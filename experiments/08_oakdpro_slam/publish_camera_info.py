"""Republish CameraInfo alongside cam0 images (the bag contract has none).

Reads pinhole intrinsics from the rig yaml (sim bags: exact GT calibration)
or from a calib_<serial>.yaml written by hw/record_oak.py (--calib), and
publishes a stamped CameraInfo for every /uav1/cam0/color/image_raw message
so rtabmap's synchronizers line up. ROS1/rospy — runs inside 3dfe/rtabmap.
"""

import argparse

import rospy
import yaml
from sensor_msgs.msg import CameraInfo, Image


def load_intrinsics(path: str, from_calib: bool):
    c = yaml.safe_load(open(path))
    if from_calib:
        cc = c["cameras"][0]
        return cc["width"], cc["height"], cc["fx"], cc["fy"], cc["cx"], cc["cy"]
    cam = c["cameras"][0]
    intr = cam["intrinsics"]
    w, h = cam["resolution"]
    return w, h, intr["fx"], intr["fy"], intr["cx"], intr["cy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rig", required=True, help="rig yaml or calib yaml path")
    ap.add_argument("--calib", action="store_true",
                    help="--rig points at a calib_<serial>.yaml (HW bag)")
    args = ap.parse_args(rospy.myargv()[1:])

    w, h, fx, fy, cx, cy = load_intrinsics(args.rig, args.calib)
    info = CameraInfo()
    info.width, info.height = w, h
    info.distortion_model = "plumb_bob"
    info.D = [0.0] * 5      # rectified/sim-exact pinhole by contract
    info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]

    rospy.init_node("cam0_info_publisher")
    pub = rospy.Publisher("/uav1/cam0/camera_info", CameraInfo, queue_size=10)

    def on_image(msg: Image):
        info.header = msg.header
        pub.publish(info)

    rospy.Subscriber("/uav1/cam0/color/image_raw", Image, on_image, queue_size=10)
    rospy.spin()


if __name__ == "__main__":
    main()
