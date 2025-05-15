import numpy as np
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as RR


def create_coor(img, pose, intrinsic_matrix,corners_3D_size = 0.1,line_size=2,):
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
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), x_color, line_size)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[2]), y_color, line_size)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), z_color, line_size)

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

def transform_poses_source(source_in_target, target_in_camera):

    T_source_in_camera = np.dot(target_in_camera, source_in_target)

    return np.array(T_source_in_camera)

def normalize_workspace(workspace, pose):
    ''' workspace: [T, 3(xyz max) + 3(xyz min) + 3(rxryrz max)+ 3(rxryrz min)]'''
    TRANS_MAX = np.array([workspace[:, 0].max(),workspace[:, 1 ].max(),workspace[:, 2 ].max()])
    TRANS_MIN = np.array([workspace[:, 3 ].min(),workspace[:, 4 ].min(),workspace[:, 5 ].min()])


    pose[:3,3] = (pose[:3,3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
    return pose

def unnormalize_action(workspace, pose):
    ''' workspace: [T, 3(xyz max) + 3(xyz min) + 3(rxryrz max)+ 3(rxryrz min)]'''
    TRANS_MAX = np.array([workspace[:, 0].max(),workspace[:, 1 ].max(),workspace[:, 2 ].max()])
    TRANS_MIN = np.array([workspace[:, 3 ].min(),workspace[:, 4 ].min(),workspace[:, 5 ].min()])

    pose[:3,3]  = (pose[:3,3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    return pose





cam_in_base = np.identity(4)
cam_in_base[:3, :3] = RR.from_euler("xyz", np.array([-88.281, 2.101, -89.142]), degrees=True).as_matrix()
cam_in_base[:3, 3] = np.array([-0.127, 0.292, -0.040])

source_workspace = np.load('./source_workspace.npy')


i = 0
data_path = '/home/sunh/1RobotMPL/HUAWEI/train'
source_pose = np.load(data_path + '/{}/source_pose.npy'.format(i)) #source_pose
target_pose = np.load(data_path + '/{}/target_pose.npy'.format(i)) #target_pose
color_all = np.load(data_path + '/{}/color_all.npy'.format(i))


fx = 455.15264892578125
px = 327.128662109375
fy = 455.15264892578125
py = 240.3665771484375
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])


gripper_in_sources = []
source_in_targets = []

for i in range(0,len(source_pose)):
    color = color_all[i]

    source_action = source_pose[i:i+10]
    target_action = target_pose[i:i+10]

    for idx in range(len(source_action)):
        if idx%5 != 0:
            continue

        source = pose_to_matrix(source_action[idx])
        target = pose_to_matrix(target_action[idx])

        print(source)

        ## sym
        # target[:3, 0] = np.array([1,0,0])
        # z = target[:3,2]
        # target[:3,1] = np.cross(z,  target[:3, 0])
        # target[:3,1] = target[:3,1] / np.linalg.norm(target[:3,1])
        ## sym

        source_in_target = np.dot(np.linalg.inv(target),source)
        source_in_target = normalize_workspace(source_workspace, source_in_target)
        source_in_target = unnormalize_action(source_workspace, source_in_target)
        source_in_camera = transform_poses_source(source_in_target, target)

        color = create_coor(color,source_in_camera[:3,:4], intrinsic_matrix, corners_3D_size = 0.02, line_size=2)
        color = create_coor(color,target[:3,:4], intrinsic_matrix, corners_3D_size = 0.09)

    cv2.imshow('s',color)
    cv2.waitKey(0)



print("source_in_target XYZ max : {}, {} , {}".format(np.array(source_in_targets)[:,0,3].max(),
                                                       np.array(source_in_targets)[:,1,3].max(),
                                                       np.array(source_in_targets)[:,2,3].max()))
print("source_in_target XYZ min : {}, {} , {}".format(np.array(source_in_targets)[:,0,3].min(),
                                                       np.array(source_in_targets)[:,1,3].min(),
                                                       np.array(source_in_targets)[:,2,3].min()))


