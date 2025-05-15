#!/usr/bin/env python
import time

import cv2
import numpy as np
from sensor_msgs.msg import Image as IMG
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

from sensor_msgs.msg import JointState

import rospy
from std_msgs.msg import Float64MultiArray, Header
from scipy.spatial.transform import Rotation as RR
from geometry_msgs.msg import PoseStamped,TransformStamped


import tf2_ros
import tf.transformations as tf_trans
from bimanual import tool_forward_kinematics


def create_coor(img, pose, intrinsic_matrix,corners_3D_size = 0.1):
    "Create a bounding box around the object"
    # 8 corner points of the ptcld data

    x_color = (255, 0, 0)
    y_color = (0, 255, 0)
    z_color = (0, 0, 255)
    corners_3D = np.array([[0, 0, 0],
                           [corners_3D_size, 0, 0],
                           [0, corners_3D_size, 0],
                           [0, 0, corners_3D_size]  ])

    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)

    # Perspective Projection to obtain 2D coordinates for masks
    homogenous_2D = intrinsic_matrix @ (pose @ homogenous_coordinate.T)
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)

    print(coord_2D)

    # Draw lines between these 8 points
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), x_color, 2)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[2]), y_color, 2)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), z_color, 2)

    return img

def fpose_to_matrix(pose):
    x, y, z, qx, qy, qz, qw = pose
    r = RR.from_quat(np.array([qx, qy, qz, qw]))
    mat = r.as_matrix()
    T_mat = np.identity(4)
    T_mat[:3, :3] = mat
    T_mat[:3, 3] = np.array([x, y, z])
    return T_mat



class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)
        self.bridge = CvBridge()


        rospy.Subscriber("/puppet/joint_left", JointState, self.joint_right_callback)   #/puppet/end_right  /puppet/joint_right
        self.rgb_sub = Subscriber("/camera_f/color/image_raw", IMG) # camera_f  camera_l
        self.depth_sub = Subscriber("/camera_f/depth/image_raw", IMG)
        self.astt = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1000, slop=0.1)
        self.astt.registerCallback(self.image_callback)

        self.cam_in_base = np.identity(4)

        # self.cam_in_base[:3, :3] = RR.from_euler("xyz", np.array([ -88.281, 2.101, -89.142]), degrees=True).as_matrix()
        # self.cam_in_base[:3, 3] = np.array([ -0.127, 0.292, -0.040 ])
        self.cam_in_base[:3, :3] = RR.from_euler("xyz", np.array([ -89.638, 0.830, -89.264]), degrees=True).as_matrix()
        self.cam_in_base[:3, 3] = np.array([ -0.130, -0.314, -0.042 ])


        fx = 455.15264892578125
        px = 327.128662109375
        fy = 455.15264892578125
        py = 240.3665771484375
        self.intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])


    def image_callback(self, rgb_msg, depth_msg):
        try:
            # 转换图像并放入队列以便后续处理
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgra8")
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1") # Y11  16UC1
        except CvBridgeError as e:
            print(e)

    def joint_right_callback(self, msg):
        # print(msg)
        self.joint_state = msg.position




    def process_images(self):


        ee_in_base_ori = tool_forward_kinematics(self.joint_state[:6])
        ee_in_base = np.identity(4)
        ee_in_base[:3,3] = ee_in_base_ori[:3]
        ee_in_base[:3,:3] = RR.from_euler("xyz",ee_in_base_ori[3:],degrees=False).as_matrix()

        gg_in_ee = np.identity(4)
        gg_in_ee[0,3] = 0.145
        gg_in_base = np.dot(ee_in_base,gg_in_ee)



        gg_in_cam = np.dot(np.linalg.inv(self.cam_in_base),gg_in_base)
        ee_in_cam = np.dot(np.linalg.inv(self.cam_in_base),ee_in_base)

        print(gg_in_cam)

        color = create_coor(self.rgb_image, gg_in_cam[:3, :4], self.intrinsic_matrix)

        cv2.imshow('gg_in_cam',color)
        cv2.waitKey(1)

    def spin(self):
        time.sleep(2)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.process_images()
            rate.sleep()


if __name__ == '__main__':
    processor = ImageProcessor()
    try:
        processor.spin()
    except KeyboardInterrupt:
        print("shutting down")
    cv2.destroyAllWindows()



