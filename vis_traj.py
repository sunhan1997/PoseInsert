import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as RR
import cv2
import numpy as np
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as RR


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





fx = 455.15264892578125
px = 327.128662109375
fy = 455.15264892578125
py = 240.3665771484375
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

trag_num = 7

source_in_target_trajectories_all = []
for idx in range(trag_num):
    source_in_target_trajectories =[]
    source_pose = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/source_pose.npy'.format(idx))
    target_pose = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/target_pose.npy'.format(idx))
    color_all = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/color_all.npy'.format(idx))
    color = color_all[0]

    source_action = source_pose[0:len(source_pose)]
    target_action = target_pose[0:len(source_pose)]
    for j in range(len(source_action)):
        source = pose_to_matrix(source_action[j])
        target = pose_to_matrix(target_action[j])

        source_in_target = np.dot(np.linalg.inv(target), source)
        # source_in_target = normalize_workspace(source_workspace, source_in_target)

        source_in_target_trajectories.append(source_in_target)
        # source_in_target = unnormalize_action(source_workspace, source_in_target)

        source_in_camera = transform_poses_source(source_in_target, target)
        color = create_coor(color, source_in_camera[:3, :4], intrinsic_matrix, corners_3D_size=0.02)
        color = create_coor(color, target[:3, :4], intrinsic_matrix)

    source_in_target_trajectories = np.array(source_in_target_trajectories)
    source_in_target_trajectories_all.append(source_in_target_trajectories)


#################3
trajectories = []
for i in range(trag_num):  # 遍历 7 条轨迹
    trajectory = source_in_target_trajectories_all[i]  # 提取平移部分 (x, y, z)
    trajectory = trajectory[:, :3, 3]  # 提取平移部分 (x, y, z)
    trajectories.append(trajectory)

# 创建 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义颜色列表
colors = ['b', 'b', 'b', 'b', 'b', 'b', 'b', ]

# 绘制每条轨迹
for i, trajectory in enumerate(trajectories):
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=colors[i], label=f'Trajectory {i + 1}')


# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('7 Source in Target Trajectories Visualization')
ax.legend()

# 显示图形
plt.show()
