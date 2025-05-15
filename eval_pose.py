import cv2
import torch
import argparse

from copy import deepcopy
from easydict import EasyDict as edict
from utils.training import set_seed
import time
import numpy as np
from dataset.pose_data import matrix_to_pose,transform_poses,sym_process
from scipy.spatial.transform import Rotation as RR
from policy.policy import PoseDP



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

    # Draw lines between these 8 points
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), x_color, 2)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[2]), y_color, 2)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), z_color, 2)

    return img

def pose_to_matrix(pose):
    """
    将位姿 (x, y, z, qx, qy, qz, qw) 转换为 4x4 的齐次变换矩阵
    :param pose: 位姿，形状为 (7,)
    :return: 4x4 的齐次变换矩阵
    """
    x, y, z, qx, qy, qz, qw = pose
    r = RR.from_quat(np.array([qx, qy, qz, qw]))
    mat = r.as_matrix()
    T_mat = np.identity(4)
    T_mat[:3,:3] = mat
    T_mat[:3,3] = np.array([x, y, z])
    return T_mat


def transform_pose(source_in_camera, target_in_camera):
    T_camera_in_target = np.linalg.inv(target_in_camera)
    T_source_in_target = np.dot(T_camera_in_target, source_in_camera)

    pose_source_in_target = matrix_to_pose(T_source_in_target)
    return pose_source_in_target





def transform_poses_source(source_in_target, target_in_camera):

    T_source_in_camera = np.dot(target_in_camera, source_in_target)

    return np.array(T_source_in_camera)

# def get_gripper_in_cam(gripper_in_base):
#     cam_in_base = np.identity(4)
#     cam_in_base[:3, :3] = RR.from_euler("xyz", np.array([-88.281, 2.101, -89.142]), degrees=True).as_matrix()
#     cam_in_base[:3, 3] = np.array([-0.127, 0.292, -0.040])
#
#     base_in_cam = np.linalg.inv(cam_in_base)
#     gripper_in_camera = np.dot(base_in_cam,gripper_in_base)
#
#     return gripper_in_camera


def normalize_workspace(workspace, pose):
    ''' workspace: [T, 3(xyz max) + 3(xyz min) + 3(rxryrz max)+ 3(rxryrz min)]'''
    TRANS_MAX = np.array([workspace[:, 0].max(),workspace[:, 1 ].max(),workspace[:, 2 ].max()])
    TRANS_MIN = np.array([workspace[:, 3 ].min(),workspace[:, 4 ].min(),workspace[:, 5 ].min()])

    print(TRANS_MAX)
    print(TRANS_MIN)

    pose[:3] = (pose[:3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
    return pose

def unnormalize_action(workspace, pose):
    ''' workspace: [T, 3(xyz max) + 3(xyz min) + 3(rxryrz max)+ 3(rxryrz min)]'''
    TRANS_MAX = np.array([workspace[:, 0].max(),workspace[:, 1 ].max(),workspace[:, 2 ].max()])
    TRANS_MIN = np.array([workspace[:, 3 ].min(),workspace[:, 4 ].min(),workspace[:, 5 ].min()])

    pose[:3,3]  = (pose[:3,3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    return pose


default_args = edict({
    "ckpt": None,
    "calib": "calib/",
    "num_action": 20,
    "num_inference_step": 20,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "max_steps": 300,
    "seed": 233,
    "vis": False,
    "discretize_rotation": True,
    "ensemble_mode": "act"
})

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', action='store', type=str, help='checkpoint path', required=False,
                    default='./logs/mp/policy_epoch_1200_seed_233.ckpt')
parser.add_argument('--calib', action='store', type=str, help='calibration path', required=False,
                    default='/home/robotlab/sunhan/RISE/data/push_block/calib/1706769947798')
parser.add_argument('--num_action', action='store', type=int, help='number of action steps', required=False,
                    default=20)
parser.add_argument('--num_inference_step', action='store', type=int, help='number of inference query steps',
                    required=False,
                    default=4)
parser.add_argument('--voxel_size', action='store', type=float, help='voxel size', required=False, default=0.005)
parser.add_argument('--obs_feature_dim', action='store', type=int, help='observation feature dimension',
                    required=False, default=512)
parser.add_argument('--hidden_dim', action='store', type=int, help='hidden dimension', required=False, default=512)
parser.add_argument('--nheads', action='store', type=int, help='number of heads', required=False, default=8)
parser.add_argument('--num_encoder_layers', action='store', type=int, help='number of encoder layers',
                    required=False, default=4)
parser.add_argument('--num_decoder_layers', action='store', type=int, help='number of decoder layers',
                    required=False, default=1)
parser.add_argument('--dim_feedforward', action='store', type=int, help='feedforward dimension', required=False,
                    default=2048)
parser.add_argument('--dropout', action='store', type=float, help='dropout ratio', required=False, default=0.1)
parser.add_argument('--max_steps', action='store', type=int, help='max steps for evaluation', required=False,
                    default=300)
parser.add_argument('--seed', action='store', type=int, help='seed', required=False, default=233)
parser.add_argument('--vis', action='store_true', help='add visualization during evaluation', default=True)
parser.add_argument('--discretize_rotation', action='store_true', help='whether to discretize rotation process.')
parser.add_argument('--ensemble_mode', action='store', type=str, help='temporal ensemble mode', required=False,
                    default='act')


fx = 455.15264892578125
px = 327.128662109375
fy = 455.15264892578125
py = 240.3665771484375
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
args_override = vars(parser.parse_args())




args = deepcopy(default_args)
for key, value in args_override.items():
    args[key] = value

# set up device
set_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# policy
print("Loading policy ...")
policy = PoseDP(
    num_action=20,
    input_dim=9,
    obs_feature_dim=128,
    action_dim=9,
    hidden_dim=512,
).to(device)

n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

# load checkpoint
assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
print("Checkpoint {} loaded.".format(args.ckpt))




i = 0
normalize = True
source_workspace = np.load('./source_workspace.npy')
# gripper_workspace = np.load('./gripper_workspace.npy')

source_poses = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/source_pose.npy'.format(i)) #cube source_pose
target_poses = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/target_pose.npy'.format(i)) #screw target_pose
color_all = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/color_all.npy'.format(i))

for idx in range(0,len(source_poses)):
    source_pose = source_poses[idx]
    target_pose = target_poses[idx]
    source_pose = pose_to_matrix(source_pose)
    target_pose = pose_to_matrix(target_pose)
    # target_pose = sym_process(target_pose)
    source_in_target = transform_pose(source_pose, target_pose)


    if normalize:
        source_in_target = normalize_workspace(source_workspace, source_in_target)


    # 进行图像处理
    color =color_all[idx]

    with torch.inference_mode():
        policy.eval()


        pose = np.array(source_in_target)
        pose = pose_to_matrix(pose)[:3,[0, 1, 3]]
        pose = torch.from_numpy(np.array([pose])).float()
        pose = pose.unsqueeze(0).to(device)

        # predict
        start =time.time()
        actions = policy(pose, actions=None, batch_size=1).squeeze(0).cpu().numpy()

        print(time.time()-start)
        source_in_target_action = actions[:, :9]


        if args.vis:
            from dataset.pose_data import pose_to_matrix

            # for i in range(source_in_target_action.shape[0]):
            for i in range(20):
                source_in_target  = source_in_target_action[i]



                source_in_target = source_in_target.reshape(3, 3)
                T_source_in_target = np.identity(4)
                x = source_in_target[:, 0]
                y = source_in_target[:, 1]
                z = np.cross(x, y)
                T_source_in_target[:3, :2] = source_in_target[:3, :2]
                T_source_in_target[:3, 2] = z
                T_source_in_target[:3, 3] = source_in_target[:3, 2]
                T_source_in_target = unnormalize_action(source_workspace, T_source_in_target)
                source_in_camera = transform_poses_source(T_source_in_target, target_pose)


                source_pose_mat = source_pose
                target_pose_mat = target_pose

                color = create_coor(color, source_in_camera[:3, :4], intrinsic_matrix, corners_3D_size=0.02)  #  source_in_camera
                color = create_coor(color, source_pose_mat[:3, :4], intrinsic_matrix)
                color = create_coor(color, target_pose_mat[:3, :4], intrinsic_matrix)
            cv2.imshow('s', color)
            cv2.waitKey(0)



