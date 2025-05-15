#!/home/lin/software/miniconda3/envs/aloha/bin/python
#coding=utf-8
import rospy
from geometry_msgs.msg import PoseStamped

import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
import time
from sensor_msgs.msg import JointState
from bimanual import SingleArm,tool_forward_kinematics


def lerp(start, end, t):
    """Linear interpolation between start and end by a factor of t."""
    return (1 - t) * start + t * end


def slerp(start_rot, end_rot, t):
    """Spherical linear interpolation between two rotations by a factor of t."""
    # 创建旋转对象
    rotation_start = R.from_euler('xyz',start_rot,degrees=False)
    rotation_end = R.from_euler('xyz',end_rot,degrees=False)
    # 定义关键帧的时间点
    times = [0, 1]
    # 创建包含所有关键帧旋转的对象数组
    key_rots = R.concatenate([rotation_start, rotation_end])
    # 创建 Slerp 对象
    slerp_obj = Slerp(times, key_rots)
    # 执行插值
    interpolated_rotation = slerp_obj(t)
    return interpolated_rotation.as_euler("xyz")

#
def inter_pose(current_pose, target_pose, step_size=0.005, angular_step_deg=0.01):
    current_position = np.array(current_pose[:3])
    current_orientation = np.array(current_pose[3:6])

    target_position = np.array(target_pose[:3])
    target_orientation = np.array(target_pose[3:6])


    total_translation = np.linalg.norm(target_position - current_position)
    num_translation_steps = int(np.ceil(total_translation / step_size))

    if num_translation_steps<20:
        num_steps = num_translation_steps +1
    else:
        num_steps = 80


    interpolated_trans = []
    interpolated_qua = []

    for i in range(num_steps + 1):
        t = i / num_steps
        new_position = lerp(current_position, target_position, t)
        new_orientation = slerp(current_orientation, target_orientation, t)

        interpolated_trans.append(new_position)
        interpolated_qua.append(new_orientation)

    return np.array(interpolated_trans), np.array(interpolated_qua)


class ArmController:
    def __init__(self):
        rospy.init_node('arm_controller', anonymous=True)
        arm_config: Dict[str, Any] = {
            "can_port": "can1",
            "type": 0,
            # Add necessary configuration parameters for the left arm
        }
        self.single_arm = SingleArm(arm_config)

        rospy.Subscriber("/right/pose", PoseStamped, self.right_pose_callback)
        self.pub = rospy.Publisher("/right/joi", PoseStamped, queue_size=1000)
        self.right_pose = np.array([0,0,0,0,0,0,0])



    def right_pose_callback(self, msg):
        self.right_pose = np.array( [msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z,
                                      msg.pose.orientation.x,
                                      msg.pose.orientation.y,
                                      msg.pose.orientation.z,
                                      msg.pose.orientation.w])

    def arm_move(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            self.joint_state = self.single_arm.get_joint_positions()
            ee_in_base_ori = tool_forward_kinematics(self.joint_state[:6])

            inter_trans, inter_qua = inter_pose(ee_in_base_ori, self.right_pose[:6])
            griper = self.right_pose[6]
            for pos, ori in zip(inter_trans, inter_qua):
                print("go")
                self.single_arm.set_ee_pose_xyzrpy(xyzrpy=[pos[0], pos[1], pos[2], ori[0], ori[1], ori[2]])
                self.single_arm.set_catch_pos(pos=griper)
                rate.sleep()




if __name__ == '__main__':
    arm = ArmController()
    try:
        arm.arm_move()
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")


