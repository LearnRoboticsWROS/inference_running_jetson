from setuptools import find_packages, setup

package_name = 'inference_running'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'ultralytics', 'opencv-python'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_annotator = inference_running.yolo_annotator:main',
            'pose_estimation = inference_running.pose_estimation:main',
            'pose_estimation_stable = inference_running.pose_estimation_stable:main',
        ],
    },
)
