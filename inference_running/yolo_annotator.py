#!/usr/bin/env python3
import os
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

import cv2
import numpy as np
from ultralytics import YOLO


def qos_video(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,   # IMPORTANT
        durability=QoSDurabilityPolicy.VOLATILE,
    )


class YoloAnnotator(Node):
    def __init__(self):
        super().__init__("yolo_annotator")

        self.declare_parameter("input_topic", "/camera/color/image_raw")  # BASE TOPIC
        self.declare_parameter("output_topic", "/camera/color/image_raw_annotated")
        self.declare_parameter("publish_raw", False)          # <= di default NO raw su Wi-Fi
        self.declare_parameter("publish_compressed", True)
        self.declare_parameter("jpeg_quality", 75)

        self.declare_parameter("model_path", "/models/capsules/best.pt")  # better engine
        self.declare_parameter("device", "0")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.publish_raw = bool(self.get_parameter("publish_raw").value)
        self.publish_compressed = bool(self.get_parameter("publish_compressed").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self.model_path = self.get_parameter("model_path").value
        self.device = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.get_logger().info(f"Loading model: {self.model_path} (device={self.device})")
        self.model = YOLO(self.model_path)

        self.bridge = CvBridge()
        self.qos = qos_video(depth=1)

        # Publishers
        self.pub_raw = self.create_publisher(Image, self.output_topic, self.qos)
        self.pub_comp = self.create_publisher(CompressedImage, self.output_topic + "/compressed", self.qos)

        # Subscriber: RAW Image (poi scegli transport=compressed lato viewer)
        self.sub = self.create_subscription(Image, self.input_topic, self.cb_raw, self.qos)

        self.busy = False
        self.busy_lock = threading.Lock()

        self.frames = 0
        self.t0 = time.time()

        self.get_logger().info(f"Subscribing (BASE): {self.input_topic}")
        self.get_logger().info(f"Publishing: {self.output_topic} (+ /compressed)")

    def cb_raw(self, msg: Image):
        # drop-frame policy: if you are processing, discart!
        with self.busy_lock:
            if self.busy:
                return
            self.busy = True

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            results = self.model.predict(
                source=frame,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False
            )

            annotated = results[0].plot()

            # publish compressed (suggested on Wi-Fi)
            if self.publish_compressed:
                ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if ok:
                    cmsg = CompressedImage()
                    cmsg.header = msg.header
                    cmsg.format = "jpeg"
                    cmsg.data = jpg.tobytes()
                    self.pub_comp.publish(cmsg)

            # publish raw (only if you need)
            if self.publish_raw:
                out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out_msg.header = msg.header
                self.pub_raw.publish(out_msg)

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


def main():
    rclpy.init()
    node = YoloAnnotator()
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