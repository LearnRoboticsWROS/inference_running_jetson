#!/usr/bin/env python3
import os
import time
import threading
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

import cv2
import numpy as np
from ultralytics import YOLO


def qos_video(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


class Pose3DEstimation(Node):
    def __init__(self):
        super().__init__("pose_3d_estimation")

        # Topics
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("cam_info_topic", "/camera/aligned_depth_to_color/camera_info")

        # Output
        self.declare_parameter("output_topic", "/camera/color/pose_estimation")
        self.declare_parameter("publish_raw", False)
        self.declare_parameter("publish_compressed", True)
        self.declare_parameter("jpeg_quality", 75)

        # YOLO
        self.declare_parameter("model_path", "/models/capsules/best.pt")
        self.declare_parameter("device", "cuda")     # "cuda" or "cpu"
        self.declare_parameter("device_id", 0)       # int
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)

        # Depth sampling
        self.declare_parameter("depth_unit", "auto")   # auto|mm|m
        self.declare_parameter("depth_patch", 7)       # odd
        self.declare_parameter("max_depth_m", 3.0)
        self.declare_parameter("min_depth_m", 0.08)

        # Fallback search inside bbox
        self.declare_parameter("bbox_shrink", 0.15)
        self.declare_parameter("bbox_grid_step", 6)
        self.declare_parameter("min_valid_samples", 20)

        # Read params
        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.cam_info_topic = self.get_parameter("cam_info_topic").value

        self.output_topic = self.get_parameter("output_topic").value
        self.publish_raw = bool(self.get_parameter("publish_raw").value)
        self.publish_compressed = bool(self.get_parameter("publish_compressed").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self.model_path = self.get_parameter("model_path").value

        device = str(self.get_parameter("device").value).strip().lower()
        device_id = int(self.get_parameter("device_id").value)

        if device in ("cuda", "gpu", "nvidia"):
            self.device = f"cuda:{device_id}"
        elif device in ("cpu",):
            self.device = "cpu"
        else:
            # allow power-users to pass "cuda:0" or "0" as raw string
            self.device = str(self.get_parameter("device").value).strip("'\"")

        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)

        self.depth_unit = str(self.get_parameter("depth_unit").value)
        self.depth_patch = int(self.get_parameter("depth_patch").value)
        if self.depth_patch < 1:
            self.depth_patch = 1
        if self.depth_patch % 2 == 0:
            self.depth_patch += 1

        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)

        self.bbox_shrink = float(self.get_parameter("bbox_shrink").value)
        self.bbox_grid_step = int(self.get_parameter("bbox_grid_step").value)
        self.min_valid_samples = int(self.get_parameter("min_valid_samples").value)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Model
        self.get_logger().info(f"Loading model: {self.model_path} (device={self.device})")
        self.model = YOLO(self.model_path)

        self.bridge = CvBridge()
        self.qos = qos_video(depth=1)

        # Cache
        self._depth_np: Optional[np.ndarray] = None
        self._depth_dtype = None
        self._K: Optional[np.ndarray] = None
        self._cam_frame_id: str = "camera_optical_link"

        # Publishers
        self.pub_raw = self.create_publisher(Image, self.output_topic, self.qos)
        self.pub_comp = self.create_publisher(CompressedImage, self.output_topic + "/compressed", self.qos)

        # Subscribers
        self.sub_color = self.create_subscription(Image, self.color_topic, self.cb_color, self.qos)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.cb_depth, self.qos)
        self.sub_info = self.create_subscription(CameraInfo, self.cam_info_topic, self.cb_info, self.qos)

        # drop-frame
        self.busy = False
        self.busy_lock = threading.Lock()

        self.frames = 0
        self.t0 = time.time()

        self.get_logger().info(f"Color:  {self.color_topic}")
        self.get_logger().info(f"Depth:  {self.depth_topic}")
        self.get_logger().info(f"Info:   {self.cam_info_topic}")
        self.get_logger().info(f"Output: {self.output_topic} (+/compressed)")

    def cb_info(self, msg: CameraInfo):
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if msg.header.frame_id:
            self._cam_frame_id = msg.header.frame_id

    def cb_depth(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warning(f"Depth cv_bridge error: {e}")
            return
        self._depth_np = depth
        self._depth_dtype = depth.dtype

    def cb_color(self, msg: Image):
        with self.busy_lock:
            if self.busy:
                return
            self.busy = True
        try:
            if self._K is None or self._depth_np is None:
                return

            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            results = self.model.predict(
                source=frame,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False
            )

            r0 = results[0]
            annotated = r0.plot()

            if r0.boxes is None or len(r0.boxes) == 0:
                self._publish(annotated, msg.header)
                return

            fx = float(self._K[0, 0]); fy = float(self._K[1, 1])
            cx = float(self._K[0, 2]); cy = float(self._K[1, 2])

            h, w = annotated.shape[:2]

            boxes_xyxy = r0.boxes.xyxy.detach().cpu().numpy()
            confs = r0.boxes.conf.detach().cpu().numpy()
            clss = r0.boxes.cls.detach().cpu().numpy().astype(int) if r0.boxes.cls is not None else None
            names = r0.names if hasattr(r0, "names") else None

            for i, xyxy in enumerate(boxes_xyxy):
                x1, y1, x2, y2 = xyxy.tolist()
                u = int(round((x1 + x2) * 0.5))
                v = int(round((y1 + y2) * 0.5))
                u = max(0, min(w - 1, u))
                v = max(0, min(h - 1, v))

                Z = self._depth_at_center_or_bbox(x1, y1, x2, y2, u, v)
                if Z is None:
                    cv2.circle(annotated, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(annotated, "Depth invalid", (u + 5, v - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                    continue

                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy

                label = ""
                if names is not None and clss is not None:
                    label = names.get(int(clss[i]), "")
                c = float(confs[i])

                cv2.circle(annotated, (u, v), 5, (0, 255, 255), -1)
                cv2.putText(
                    annotated,
                    f"{label} {c:.2f} XYZ:[{X:+.3f},{Y:+.3f},{Z:+.3f}]m",
                    (max(0, int(x1)), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA
                )

            cv2.putText(
                annotated, f"frame: {self._cam_frame_id}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
            )

            self._publish(annotated, msg.header)

            self.frames += 1
            if self.frames % 30 == 0:
                dt = time.time() - self.t0
                fps = self.frames / dt if dt > 0 else 0.0
                self.get_logger().info(f"FPS(processed): {fps:.2f}")

        except Exception as e:
            self.get_logger().warning(f"processing error: {e}")
        finally:
            with self.busy_lock:
                self.busy = False

    # ---- depth helpers ----
    def _to_meters(self, arr: np.ndarray) -> np.ndarray:
        unit = self.depth_unit
        if unit == "auto":
            unit = "mm" if self._depth_dtype == np.uint16 else "m"
        if unit == "mm":
            return arr * 0.001
        return arr

    def _depth_median_patch(self, u: int, v: int) -> Optional[float]:
        depth = self._depth_np
        if depth is None:
            return None
        dh, dw = depth.shape[:2]
        r = self.depth_patch // 2
        u0 = max(0, u - r); u1 = min(dw - 1, u + r)
        v0 = max(0, v - r); v1 = min(dh - 1, v + r)

        patch = depth[v0:v1+1, u0:u1+1].astype(np.float32)
        patch = self._to_meters(patch)
        patch = patch[np.isfinite(patch)]
        patch = patch[(patch > self.min_depth_m) & (patch < self.max_depth_m)]
        if patch.size == 0:
            return None
        return float(np.median(patch))

    def _depth_from_bbox(self, x1, y1, x2, y2) -> Optional[float]:
        depth = self._depth_np
        if depth is None:
            return None
        dh, dw = depth.shape[:2]

        x1 = max(0, min(dw - 1, int(x1))); x2 = max(0, min(dw - 1, int(x2)))
        y1 = max(0, min(dh - 1, int(y1))); y2 = max(0, min(dh - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None

        sx = int((x2 - x1) * self.bbox_shrink)
        sy = int((y2 - y1) * self.bbox_shrink)
        x1s, x2s = x1 + sx, x2 - sx
        y1s, y2s = y1 + sy, y2 - sy
        if x2s <= x1s or y2s <= y1s:
            x1s, x2s, y1s, y2s = x1, x2, y1, y2

        step = max(2, self.bbox_grid_step)
        vals: List[float] = []

        for vv in range(y1s, y2s, step):
            row = depth[vv, x1s:x2s:step].astype(np.float32)
            row = self._to_meters(row)
            row = row[np.isfinite(row)]
            row = row[(row > self.min_depth_m) & (row < self.max_depth_m)]
            if row.size > 0:
                vals.extend(row.tolist())

        if len(vals) < self.min_valid_samples:
            return None
        return float(np.median(np.array(vals, dtype=np.float32)))

    def _depth_at_center_or_bbox(self, x1, y1, x2, y2, u, v) -> Optional[float]:
        Z = self._depth_median_patch(u, v)
        if Z is not None:
            return Z
        return self._depth_from_bbox(x1, y1, x2, y2)

    # ---- publish ----
    def _publish(self, bgr_img: np.ndarray, header):
        if self.publish_compressed:
            ok, jpg = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                cmsg = CompressedImage()
                cmsg.header = header
                cmsg.format = "jpeg"
                cmsg.data = jpg.tobytes()
                self.pub_comp.publish(cmsg)

        if self.publish_raw:
            out_msg = self.bridge.cv2_to_imgmsg(bgr_img, encoding="bgr8")
            out_msg.header = header
            self.pub_raw.publish(out_msg)


def main():
    rclpy.init()
    node = Pose3DEstimation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()