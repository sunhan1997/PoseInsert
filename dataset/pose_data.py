from torch.utils.data import Dataset
import collections.abc as container_abcs
from scipy.spatial.transform import Rotation as RR
from  policy.Utils import *


def poses_to_matrix(poses):
    """
    将位姿 (x, y, z, qx, qy, qz, qw) 转换为 4x4 的齐次变换矩阵
    :param pose: 位姿，形状为 (7,)
    :return: 4x4 的齐次变换矩阵
    """
    mats = []
    for i in range(len(poses)):
        mats.append(pose_to_matrix(poses[i]))
    return np.array(mats)


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
    T_mat[:3, :3] = mat
    T_mat[:3, 3] = np.array([x, y, z])
    return T_mat


def matrix_to_pose(matrix):
    """
    将 4x4 的齐次变换矩阵转换为位姿 (x, y, z, qx, qy, qz, qw)
    :param matrix: 4x4 的齐次变换矩阵
    :return: 位姿，形状为 (7,)
    """
    translation = matrix[:3, 3]

    r = RR.from_matrix(matrix[:3, :3])
    quaternion = r.as_quat()
    return np.concatenate([translation, quaternion])


def transform_poses(source_in_camera, target_in_camera):
    """
    将 source_in_camera 和 target_in_camera 的位姿转换为 source_in_target 的位姿
    :param source_in_camera: 形状为 (345, 7) 的数组，表示 source 在相机坐标系下的位姿
    :param target_in_camera: 形状为 (345, 7) 的数组，表示 target 在相机坐标系下的位姿
    :return: 形状为 (345, 7) 的数组，表示 source 在 target 坐标系下的位姿
    """
    source_in_target = []
    for i in range(source_in_camera.shape[0]):
        # T_source_in_camera = pose_to_matrix(source_in_camera[i])
        # T_target_in_camera = pose_to_matrix(target_in_camera[i])
        T_source_in_camera = source_in_camera[i]
        T_target_in_camera = target_in_camera[i]

        T_camera_in_target = np.linalg.inv(T_target_in_camera)
        T_source_in_target = np.dot(T_camera_in_target, T_source_in_camera)

        pose_source_in_target = matrix_to_pose(T_source_in_target)
        source_in_target.append(pose_source_in_target)
    return np.array(source_in_target)


def sym_process(pose, sym='z'):
    x = pose[:3, 0]
    y = pose[:3, 1]
    z = pose[:3, 2]
    if sym == 'z':
        tmp_x = np.array([0.06297127, -0.05639809, -0.99642053])
        tmp_y = np.cross(z, tmp_x)
        tmp_y = tmp_y / np.linalg.norm(tmp_y)
        pose[:3, 0] = tmp_x
        pose[:3, 1] = tmp_y
        return pose


def sym_process_all(poses):
    for i in range(len(poses)):
        poses[i] = sym_process(poses[i])
    return poses






def normalize_workspace(workspace, pose):
    ''' workspace: [T, 3(xyz max) + 3(xyz min) + 3(rxryrz max)+ 3(rxryrz min)]'''
    TRANS_MAX = np.array([workspace[:, 0].max(), workspace[:, 1].max(), workspace[:, 2].max()])
    TRANS_MIN = np.array([workspace[:, 3].min(), workspace[:, 4].min(), workspace[:, 5].min()])

    pose[:, :3] = (pose[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
    return pose


class RealWorldDataset_Pose(Dataset):
    """
    Real-world Dataset with gripper
    """

    def __init__(
            self,
            path,
            split='train',
            num_obs=1,
            num_action=20,
            gripper=True,
            sym_or=False,
            sym='z',
            aug=False,
            normalize=False,
    ):
        assert split in ['train', 'val', 'all']

        self.sym_or = sym_or
        self.gripper = gripper
        self.sym = sym
        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        self.num_obs = num_obs
        self.num_action = num_action

        self.all_demos = sorted(os.listdir(self.data_path))
        self.num_demos = len(self.all_demos)

        self.data_paths = []
        self.source_obs_poses = []
        self.gripper_obs_poses = []
        self.action_poses_s = []
        self.action_poses_g = []
        self.action_poses_gw = []

        source_workspace = np.load('./source_workspace.npy')
        if self.gripper:
            gripper_workspace = np.load('./gripper_workspace.npy')

        for idx in range(self.num_demos):

            source_pose = np.load(self.data_path + '/{}/source_pose.npy'.format(idx))  # target_pose
            target_pose = np.load(self.data_path + '/{}/target_pose.npy'.format(idx))  # source_pose
            source_pose = poses_to_matrix(source_pose)
            target_pose = poses_to_matrix(target_pose)
            if self.sym_or:
                target_pose = sym_process_all(target_pose)

            source_in_target = transform_poses(source_pose, target_pose)
            if normalize:
                source_in_target = normalize_workspace(source_workspace, source_in_target)

            if self.gripper:
                gripper_pose = np.load(self.data_path + '/{}/ee_pose.npy'.format(idx))
                gripper_in_cam = get_gripper_in_cam(gripper_pose)
                gripper_in_source = transform_poses(gripper_in_cam, source_pose)
                if normalize:
                    gripper_in_source = normalize_workspace(gripper_workspace, gripper_in_source)

                gripper_w = np.load(self.data_path + '/{}/gripper.npy'.format(idx)) / 4.0

            pose_ids = [i for i in range(1, len(source_in_target))]

            for cur_idx in range(len(pose_ids) - 1):
                obs_pad_before = max(0, num_obs - cur_idx - 1)
                action_pad_after = max(0, num_action - (len(pose_ids) - 1 - cur_idx))
                frame_begin = max(0, cur_idx - num_obs + 1)
                frame_end = min(len(pose_ids), cur_idx + num_action + 1)

                obs_pose_ids = pose_ids[:1] * obs_pad_before + pose_ids[frame_begin: cur_idx + 1]
                action_pose_ids = pose_ids[cur_idx + 1: frame_end] + pose_ids[-1:] * action_pad_after  # ids / pose

                source_obs_poses = source_in_target[obs_pose_ids]
                if self.gripper:
                    gripper_obs_poses = gripper_in_source[obs_pose_ids]
                    self.gripper_obs_poses.append(gripper_obs_poses)
                    action_poses_s = source_in_target[action_pose_ids]
                    action_poses_g = gripper_in_source[action_pose_ids]
                    action_poses_gw = gripper_w[action_pose_ids]

                    self.action_poses_g.append(action_poses_g)
                    self.action_poses_gw.append(action_poses_gw)
                else:
                    action_poses_s = source_in_target[action_pose_ids]

                self.source_obs_poses.append(source_obs_poses)
                self.action_poses_s.append(action_poses_s)

    def __len__(self):
        return len(self.source_obs_poses)

    def __getitem__(self, index):

        obs_source_pose = self.source_obs_poses[index]
        action_source_pose = self.action_poses_s[index]


        obs_source_pose = poses_to_matrix(obs_source_pose)[:, :3, [0, 1, 3]]
        action_source_pose = poses_to_matrix(action_source_pose)[:, :3, [0, 1, 3]]

        if self.gripper:
            action_gripper_pose = self.action_poses_g[index]
            action_gripper_width = self.action_poses_gw[index]
            action_gripper_pose = poses_to_matrix(action_gripper_pose)[:, :3, :4]
            action_gripper_pose = torch.from_numpy(action_gripper_pose).float()
            action_gripper_width = torch.from_numpy(action_gripper_width).float()

        obs_source_pose = torch.from_numpy(obs_source_pose).float()
        action_source_pose = torch.from_numpy(action_source_pose).float()

        if self.gripper:
            obs_gripper_pose = self.gripper_obs_poses[index]
            obs_gripper_pose = poses_to_matrix(obs_gripper_pose)[:, :3, :4]
            obs_gripper_pose = torch.from_numpy(obs_gripper_pose).float()
            ret_dict = {
                'obs_source_pose': obs_source_pose,
                'obs_gripper_pose': obs_gripper_pose,
                'action_source_pose': action_source_pose,
                'action_gripper_pose': action_gripper_pose,
                'action_gripper_width': action_gripper_width,
            }
        else:
            ret_dict = {
                'obs_source_pose': obs_source_pose,
                'obs_gripper_pose': None,
                'action_source_pose': action_source_pose,
                'action_gripper_pose': None,
                'action_gripper_width': None,
            }

        return ret_dict





class RealWorldDataset_PoseRGBD(Dataset):
    """
    Real-world Dataset with gripper
    """

    def __init__(
            self,
            path,
            split='train',
            num_obs=1,
            num_action=20,
            gripper=True,
            sym_or=False,
            sym='z',
            aug=False,
            normalize=False,
    ):
        assert split in ['train', 'val', 'all']

        self.sym_or = sym_or
        self.gripper = gripper
        self.sym = sym
        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        self.num_obs = num_obs
        self.num_action = num_action

        self.all_demos = sorted(os.listdir(self.data_path))
        self.num_demos = len(self.all_demos)

        self.data_paths = []
        self.source_obs_poses = []
        self.color_obs_poses = []
        self.depth_obs_poses = []
        self.gripper_obs_poses = []
        self.action_poses_s = []
        self.action_poses_g = []
        self.action_poses_gw = []

        self.source_bbox = []
        self.target_bbox = []


        source_workspace = np.load('./source_workspace.npy')
        if self.gripper:
            gripper_workspace = np.load('./gripper_workspace.npy')

        number = np.load(self.data_path + '/{}/source_pose.npy'.format(0))
        goal_color = np.load(self.data_path + '/{}/color_all.npy'.format(0))
        goal_depth = np.load(self.data_path + '/{}/depth_all.npy'.format(0))
        source_bbox = np.load(self.data_path + '/{}/source_bbox.npy'.format(0))
        target_bbox = np.load(self.data_path + '/{}/target_bbox.npy'.format(0))

        self.goal_color = goal_color[len(number) - 1]
        self.goal_depth = goal_depth[len(number) - 1] / 1000.0
        source_bbox = source_bbox[len(number) - 1]
        target_bbox = target_bbox[len(number) - 1]
        x_all = np.array(
            [source_bbox[0], source_bbox[0] + source_bbox[2], target_bbox[0], target_bbox[0] + target_bbox[2]])
        y_all = np.array(
            [source_bbox[1], source_bbox[1] + source_bbox[3], target_bbox[1], target_bbox[1] + target_bbox[3]])
        x_min, x_max = np.min(x_all), np.max(x_all)
        y_min, y_max = np.min(y_all), np.max(y_all)
        self.goal_color = self.goal_color[int(y_min):int(y_max), int(x_min):int(x_max)]
        self.goal_depth = self.goal_depth[int(y_min):int(y_max), int(x_min):int(x_max)]

        for idx in range(self.num_demos):

            source_pose = np.load(self.data_path + '/{}/source_pose.npy'.format(idx))  # target_pose
            target_pose = np.load(self.data_path + '/{}/target_pose.npy'.format(idx))  # source_pose
            color_all = np.load(self.data_path + '/{}/color_all.npy'.format(idx))  # source_pose
            depth_all = np.load(self.data_path + '/{}/depth_all.npy'.format(idx))  # source_pose

            source_bboxs = np.load(self.data_path + '/{}/source_bbox.npy'.format(idx))
            target_bboxs = np.load(self.data_path + '/{}/target_bbox.npy'.format(idx))

            source_pose = poses_to_matrix(source_pose)
            target_pose = poses_to_matrix(target_pose)
            if self.sym_or:
                target_pose = sym_process_all(target_pose)

            source_in_target = transform_poses(source_pose, target_pose)
            if normalize:
                source_in_target = normalize_workspace(source_workspace, source_in_target)

            if self.gripper:
                gripper_pose = np.load(self.data_path + '/{}/ee_pose.npy'.format(idx))
                gripper_in_cam = get_gripper_in_cam(gripper_pose)
                gripper_in_source = transform_poses(gripper_in_cam, source_pose)
                if normalize:
                    gripper_in_source = normalize_workspace(gripper_workspace, gripper_in_source)

                gripper_w = np.load(self.data_path + '/{}/gripper.npy'.format(idx)) / 4.0

            pose_ids = [i for i in range(1, len(source_in_target))]

            for cur_idx in range(len(pose_ids) - 1):
                obs_pad_before = max(0, num_obs - cur_idx - 1)
                action_pad_after = max(0, num_action - (len(pose_ids) - 1 - cur_idx))
                frame_begin = max(0, cur_idx - num_obs + 1)
                frame_end = min(len(pose_ids), cur_idx + num_action + 1)

                obs_pose_ids = pose_ids[:1] * obs_pad_before + pose_ids[frame_begin: cur_idx + 1]
                action_pose_ids = pose_ids[cur_idx + 1: frame_end] + pose_ids[-1:] * action_pad_after  # ids / pose

                source_obs_poses = source_in_target[obs_pose_ids]
                colors = color_all[obs_pose_ids]
                depths = depth_all[obs_pose_ids]
                source_bbox = source_bboxs[obs_pose_ids]
                target_bbox = target_bboxs[obs_pose_ids]

                if self.gripper:
                    gripper_obs_poses = gripper_in_source[obs_pose_ids]
                    self.gripper_obs_poses.append(gripper_obs_poses)
                    action_poses_s = source_in_target[action_pose_ids]
                    action_poses_g = gripper_in_source[action_pose_ids]
                    action_poses_gw = gripper_w[action_pose_ids]

                    self.action_poses_g.append(action_poses_g)
                    self.action_poses_gw.append(action_poses_gw)
                else:
                    action_poses_s = source_in_target[action_pose_ids]

                self.color_obs_poses.append(colors)
                self.depth_obs_poses.append(depths)
                self.source_bbox.append(source_bbox)
                self.target_bbox.append(target_bbox)
                self.source_obs_poses.append(source_obs_poses)
                self.action_poses_s.append(action_poses_s)


    def __len__(self):
        return len(self.source_obs_poses)

    def __getitem__(self, index):

        obs_color = self.color_obs_poses[index][0]
        obs_depth = self.depth_obs_poses[index][0] / 1000.0
        source_bbox = self.source_bbox[index][0]
        target_bbox = self.target_bbox[index][0]
        x_all = np.array(
            [source_bbox[0], source_bbox[0] + source_bbox[2], target_bbox[0], target_bbox[0] + target_bbox[2]])
        y_all = np.array(
            [source_bbox[1], source_bbox[1] + source_bbox[3], target_bbox[1], target_bbox[1] + target_bbox[3]])
        x_min, x_max = np.min(x_all), np.max(x_all)
        y_min, y_max = np.min(y_all), np.max(y_all)
        obs_color = obs_color[int(y_min):int(y_max), int(x_min):int(x_max)]
        obs_depth = obs_depth[int(y_min):int(y_max), int(x_min):int(x_max)]

        obs_color = cv2.resize(obs_color, (320, 320))
        obs_depth = cv2.resize(obs_depth, (320, 320))

        obs_source_pose = self.source_obs_poses[index]
        action_source_pose = self.action_poses_s[index]


        obs_source_pose = poses_to_matrix(obs_source_pose)[:, :3, [0, 1, 3]]
        action_source_pose = poses_to_matrix(action_source_pose)[:, :3, [0, 1, 3]]

        if self.gripper:
            action_gripper_pose = self.action_poses_g[index]
            action_gripper_width = self.action_poses_gw[index]
            action_gripper_pose = poses_to_matrix(action_gripper_pose)[:, :3, :4]
            action_gripper_pose = torch.from_numpy(action_gripper_pose).float()
            action_gripper_width = torch.from_numpy(action_gripper_width).float()

        # obs_rgbd = process_rgbd(obs_colors[0],obs_depths[0]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        obs_colors = torch.from_numpy(obs_color).float()
        obs_depths = torch.from_numpy(obs_depth).float()

        goal_color = torch.from_numpy(self.goal_color).float()
        goal_depth = torch.from_numpy(self.goal_depth).float()
        obs_source_pose = torch.from_numpy(obs_source_pose).float()
        action_source_pose = torch.from_numpy(action_source_pose).float()

        if self.gripper:
            obs_gripper_pose = self.gripper_obs_poses[index]
            obs_gripper_pose = poses_to_matrix(obs_gripper_pose)[:, :3, :4]
            obs_gripper_pose = torch.from_numpy(obs_gripper_pose).float()
            ret_dict = {
                'obs_source_pose': obs_source_pose,
                'obs_gripper_pose': obs_gripper_pose,
                'action_source_pose': action_source_pose,
                'action_gripper_pose': action_gripper_pose,
                'action_gripper_width': action_gripper_width,
            }
        else:
            ret_dict = {
                'obs_source_pose': obs_source_pose,
                'obs_gripper_pose': None,
                'obs_rgb': obs_colors,
                'obs_d': obs_depths,
                'goal_color': goal_color,
                'goal_depth': goal_depth,
                'action_source_pose': action_source_pose,
                'action_gripper_pose': None,
                'action_gripper_width': None,
            }

        return ret_dict


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in ['obs_source_pose', 'action_source_pose']:
                ret_dict[key] = torch.stack([b[key] for b in batch], 0)
            else:
                ret_dict[key] = [d[key] for d in batch]
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def collate_fn2(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in ['obs_source_pose', 'obs_rgb','obs_d', 'action_source_pose']:
                ret_dict[key] = torch.stack([b[key] for b in batch], 0)
            else:
                ret_dict[key] = [d[key] for d in batch]
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

