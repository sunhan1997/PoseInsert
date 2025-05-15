import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as RR


def pose_to_matrix(pose):
    x, y, z, qx, qy, qz, qw = pose
    r = RR.from_quat(np.array([qx, qy, qz, qw]))
    mat = r.as_matrix()
    T_mat = np.identity(4)
    T_mat[:3,:3] = mat
    T_mat[:3,3] = np.array([x, y, z])
    return T_mat

def poses_to_matrix(poses):
    mats = []
    for i in range(len(poses)):
        mats.append(pose_to_matrix(poses[i]))

    return np.array(mats)


def get_gripper_in_camera(gripper_in_base):
    cam_in_base = np.identity(4)
    cam_in_base[:3, :3] = RR.from_euler("xyz", np.array([-90, 0, -90]), degrees=True).as_matrix()
    cam_in_base[:3, 3] = np.array([-0.257, 0.308, -0.19])  # -0.129, 0.308, -0.056

    base_in_cam = np.linalg.inv(cam_in_base)
    gripper_in_camera = np.dot(base_in_cam, gripper_in_base)
    return gripper_in_camera

def sym_process(pose,sym='z'):
    x = pose[:3, 0]
    y = pose[:3, 1]
    z = pose[:3, 2]

    if sym=='z':
        tmp_x  = np.array([1, 0, 0])
        tmp_y = np.cross(z, tmp_x)
        tmp_y = tmp_y / np.linalg.norm(tmp_y)
        pose[:3, 0] = tmp_x
        pose[:3, 1] = tmp_y
        return pose

def sym_process_all(poses):
    for i in range(len(poses)):
        poses[i] = sym_process(poses[i])
    return poses

def matrix_to_pose_euler(matrix):
    translation = matrix[:3,3]

    r = RR.from_matrix(matrix[:3,:3])
    quaternion = r.as_quat()
    euler = r.as_euler("xyz")
    return np.concatenate([translation, euler])

def transform_poses(source_in_camera, target_in_camera):
    source_in_target = []
    for i in range(source_in_camera.shape[0]):
        # T_source_in_camera = pose_to_matrix(source_in_camera[i])
        # T_target_in_camera = pose_to_matrix(target_in_camera[i])
        T_source_in_camera = source_in_camera[i]
        T_target_in_camera = target_in_camera[i]

        T_camera_in_target = np.linalg.inv(T_target_in_camera)
        T_source_in_target = np.dot(T_camera_in_target, T_source_in_camera)

        pose_source_in_target = matrix_to_pose_euler(T_source_in_target)
        source_in_target.append(pose_source_in_target)
    return np.array(source_in_target)

def get_gripper_in_cam(gripper_in_base):
    cam_in_base = np.identity(4)
    cam_in_base[:3, :3] = RR.from_euler("xyz", np.array([-90, 0, -90]), degrees=True).as_matrix()
    cam_in_base[:3, 3] = np.array([-0.257, 0.308, -0.19])  # -0.129, 0.308, -0.056

    gripper_in_camera_all = []
    for i in range(gripper_in_base.shape[0]):
        T_gripper_in_base = gripper_in_base[i]

        base_in_cam = np.linalg.inv(cam_in_base)
        gripper_in_camera = np.dot(base_in_cam,T_gripper_in_base)
        gripper_in_camera_all.append(gripper_in_camera)

    return np.array(gripper_in_camera_all)


def normalize_workspace(workspace, pose):
    ''' workspace: [T, 3(xyz max) + 3(xyz min) + 3(rxryrz max)+ 3(rxryrz min)]'''
    TRANS_MAX = np.array([workspace[:, 0].max(),workspace[:, 1 ].max(),workspace[:, 2 ].max()])
    TRANS_MIN = np.array([workspace[:, 3 ].min(),workspace[:, 4 ].min(),workspace[:, 5 ].min()])

    print(TRANS_MAX)
    print(TRANS_MIN)

    pose[:, :3] = (pose[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
    return pose


gripper = False
source_workspace = []
gripper_workspace = []
gripper_w_space = []


for idx in range(7):

    source_pose = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/source_pose.npy'.format(idx)) #target_pose
    target_pose = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/target_pose.npy'.format(idx)) # source_pose

    source_pose = poses_to_matrix(source_pose)
    target_pose = poses_to_matrix(target_pose)
    # target_pose = sym_process_all(target_pose)

    source_in_target = transform_poses(source_pose, target_pose)

    source_workspace.append(
        np.array([source_in_target[:, 0].max(), source_in_target[:, 1].max(), source_in_target[:, 2].max(),  # xyz max
                  source_in_target[:, 0].min(), source_in_target[:, 1].min(), source_in_target[:, 2].min(),  # xyz min
                  source_in_target[:, 3].max(), source_in_target[:, 4].max(), source_in_target[:, 5].max(),
                  # rx ry rz max
                  source_in_target[:, 3].min(), source_in_target[:, 4].min(),
                  source_in_target[:, 5].min()]))  # rx ry rz max

    # if gripper:
    #     gripper_pose = np.load('/home/agilex/sunhan/collect_data/data/train/{}/ee_pose.npy'.format(idx))
    #     gripper_in_cam = get_gripper_in_cam(gripper_pose)
    #     gripper_in_source = transform_poses(gripper_in_cam, source_pose)
    #     gripper_w = np.load('/home/agilex/sunhan/collect_data/data/train/{}/gripper.npy'.format(idx))
    #
    #     gripper_workspace.append(
    #         np.array([gripper_in_source[:, 0].max(), gripper_in_source[:, 1].max(), gripper_in_source[:, 2].max(),
    #                   gripper_in_source[:, 0].min(), gripper_in_source[:, 1].min(), gripper_in_source[:, 2].min(),
    #                   gripper_in_source[:, 3].max(), gripper_in_source[:, 4].max(), gripper_in_source[:, 5].max(),
    #                   gripper_in_source[:, 3].min(), gripper_in_source[:, 4].min(), gripper_in_source[:, 5].min()]))
    #
    #     gripper_w_space.append(np.array([gripper_w.max(), gripper_w.min()]))



source_workspace  =  np.array(source_workspace)
np.save('./source_workspace.npy',source_workspace)
pass

