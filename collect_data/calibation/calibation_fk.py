#!/usr/bin/env python
import time
import numpy as np
from sensor_msgs.msg import Image as IMG
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber


import rospy
from std_msgs.msg import Float64MultiArray, Header
from scipy.spatial.transform import Rotation as RR
from geometry_msgs.msg import PoseStamped,TransformStamped


import tf2_ros
import tf.transformations as tf_trans
# from fk2 import fk_aloha,fk2_aloha

from bimanual import tool_forward_kinematics






def fpose_to_matrix(pose):
    x, y, z, qx, qy, qz, qw = pose
    r = RR.from_quat(np.array([qx, qy, qz, qw]))
    mat = r.as_matrix()
    T_mat = np.identity(4)
    T_mat[:3, :3] = mat
    T_mat[:3, 3] = np.array([x, y, z])
    return T_mat

def endpose_to_matrix(pose):
    x, y, z, rx, ry, rz, w = pose
    r = RR.from_euler('xyz', [rx, ry, rz ], degrees=False)
    mat = r.as_matrix()
    T_mat = np.identity(4)
    T_mat[:3,:3] = mat
    T_mat[:3,3] = np.array([x, y, z])
    return T_mat

class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)



        rospy.Subscriber("/puppet/joint_left", JointState, self.joint_right_callback)   #/puppet/end_right  /puppet/joint_right
        # rospy.Subscriber("/puppet/end_right", PoseStamped, self.ee_pose_callback)   #/puppet/end_right
        # rospy.Subscriber("/gripper/pose", PoseStamped, self.gripper_pose_callback)
        rospy.Subscriber("/gripper/pose", PoseStamped, self.gripper_pose_callback)

        self.joint_state = [-0.00553131103515625, 0.00934600830078125, 0.01506805419921875, -0.049019813537597656, -0.00553131103515625,
         0.00324249267578125]

        self.br = tf2_ros.TransformBroadcaster()


    def joint_right_callback(self, msg):
        # print(msg)
        self.joint_state = msg.position

    def ee_pose_callback(self, msg):
        self.ee_in_base_ros = np.array( [ msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z,
                                      msg.pose.orientation.x,
                                      msg.pose.orientation.y,
                                      msg.pose.orientation.z,
                                      msg.pose.orientation.w])

    def gripper_pose_callback(self, msg):
        self.ee_in_cam = np.array( [  msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z,
                                      msg.pose.orientation.x,
                                      msg.pose.orientation.y,
                                      msg.pose.orientation.z,
                                      msg.pose.orientation.w])

    def process_images(self):
        ee_in_base_ori = tool_forward_kinematics(self.joint_state[:6])
        print(ee_in_base_ori)

        ee_in_base = np.identity(4)
        ee_in_base[:3,3] = ee_in_base_ori[:3]
        ee_in_base[:3,:3] = RR.from_euler("xyz",ee_in_base_ori[3:],degrees=False).as_matrix()

        # ee_in_base_ros = endpose_to_matrix(self.ee_in_base_ros)
        ee_in_cam = fpose_to_matrix(self.ee_in_cam)
        cam_in_ee  = np.linalg.inv(ee_in_cam)
        
        
        print("cam_in_base: ",np.dot( ee_in_base ,cam_in_ee) )


        self.publish_tf(cam_in_ee, "ee_link", "cam_link")
        self.publish_tf(ee_in_base, "base_link", "ee_link")
        # self.publish_tf(ee_in_base_ros, "base_link", "ee_link_ros")







    def publish_tf(self, transform_matrix, parent_frame, child_frame):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # 设置平移部分
        t.transform.translation.x = transform_matrix[0, 3]
        t.transform.translation.y = transform_matrix[1, 3]
        t.transform.translation.z = transform_matrix[2, 3]

        # 设置旋转部分（四元数）
        quat = tf_trans.quaternion_from_matrix(transform_matrix)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        # 发布变换
        self.br.sendTransform(t)






    def spin(self):
        time.sleep(1)
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


# -0.132, 0.291, -0.039   -1.544, 0.047, -1.544  -88.438, 2.679, -88.456
# -0.127, 0.292, -0.040  -1.541, 0.037, -1.556  -88.281, 2.101, -89.142
# -0.128, 0.278, -0.061   -1.469, 0.036, -1.528  -84.179, 2.088, -87.537
