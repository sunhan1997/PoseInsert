#!/usr/bin/env python
import time
import numpy as np
from sensor_msgs.msg import Image as IMG
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

from sensor_msgs.msg import JointState

import rospy
import os
from std_msgs.msg import Float64MultiArray, Header
from scipy.spatial.transform import Rotation as RR
from geometry_msgs.msg import PoseStamped
import argparse

from easydict import EasyDict as edict
from copy import deepcopy

def save_data(idx, name,data):
    # 定义保存文件的路径
    save_dir = './data/usb/{}/'.format(idx)
    file_path = os.path.join(save_dir, '{}.npy'.format(name))

    # 检查并创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 保存 numpy 数组到指定路径
    np.save(file_path, data)
    print(f"Saved color_all to {file_path}")


class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)
        self.bridge = CvBridge()
        self.i = 1
        # 使用message_filters设置同步器
        self.rgb_sub = Subscriber("/camera_f/color/image_raw", IMG)
        self.depth_sub = Subscriber("/camera_f/depth/image_raw", IMG)
        self.astt = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1, slop=0.1)
        self.astt.registerCallback(self.image_callback)

        # self.pub = rospy.Publisher("/target/pose", PoseStamped, queue_size=1000)
        rospy.Subscriber("/target/pose", PoseStamped, self.target_pose_callback)
        rospy.Subscriber("/source/pose", PoseStamped, self.source_pose_callback)
        # rospy.Subscriber("/puppet/joint_right", JointState, self.joint_right_callback)   #/puppet/end_right

        rospy.Subscriber("/source/bbox", Float64MultiArray, self.source_bbox_callback)
        rospy.Subscriber("/target/bbox", Float64MultiArray, self.target_bbox_callback)


    def image_callback(self, rgb_msg, depth_msg):
        try:
            # 转换图像并放入队列以便后续处理
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgra8")
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1") # Y11  16UC1
        except CvBridgeError as e:
            print(e)

    def target_bbox_callback(self, msg):
        self.target_bbox = np.array( msg.data)

    def source_bbox_callback(self, msg):
        self.source_bbox =  np.array( msg.data)

    def target_pose_callback(self, msg):
        # self.target_pose = msg
        self.target_pose = np.array( [msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z,
                                      msg.pose.orientation.x,
                                      msg.pose.orientation.y,
                                      msg.pose.orientation.z,
                                      msg.pose.orientation.w])

    def source_pose_callback(self, msg):
        # self.source_pose = msg
        self.source_pose = np.array( [msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z,
                                      msg.pose.orientation.x,
                                      msg.pose.orientation.y,
                                      msg.pose.orientation.z,
                                      msg.pose.orientation.w])

    # def gripper_pose_callback(self, msg):
    #     # self.source_pose = msg
    #     self.gripper_pose = np.array( [msg.pose.position.x,
    #                                   msg.pose.position.y,
    #                                   msg.pose.position.z,
    #                                   msg.pose.orientation.x,
    #                                   msg.pose.orientation.y,
    #                                   msg.pose.orientation.z,
    #                                   msg.pose.orientation.w])

    def process_images(self):
        color_all = []
        depth_all = []
        source_pose = []
        target_pose = []
        source_bbox = []
        target_bbox = []
        i = 0
        while not rospy.is_shutdown():
            # 进行图像处理
            depth = np.asanyarray(self.depth_image)
            color = np.asanyarray(self.rgb_image)
            color = color[:, :, :3]
            depth_full = np.zeros((480,640))
            depth_full[:400,:] = depth

            color_all.append(color)
            depth_all.append(depth_full)
            source_pose.append(self.source_pose)
            target_pose.append(self.target_pose)

            source_bbox.append(self.source_bbox)
            target_bbox.append(self.target_bbox)

            time.sleep(0.03)

            print(i)
            i = i+1
        idx = args.idx
        save_data(idx,"color_all", color_all)
        save_data(idx,"depth_all", depth_all)
        save_data(idx,"source_pose", source_pose)
        save_data(idx,"target_pose", target_pose)

        save_data(idx,"source_bbox", source_bbox)
        save_data(idx,"target_bbox", target_bbox)










    def spin(self):
        time.sleep(1)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if hasattr(self, "rgb_image") and hasattr(self, "depth_image"):
                self.process_images()
            rate.sleep()


if __name__ == '__main__':
    default_args = edict({
        "idx": '0',
    })
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', default='0')
    args_override = vars(parser.parse_args())

    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    processor = ImageProcessor()
    try:
        processor.spin()
    except KeyboardInterrupt:
        print("shutting down")


