import cv2
from copy import deepcopy
from easydict import EasyDict as edict
from utils.training import set_seed
from policy.policy import PoseRGBD_DP
import time
import numpy as np
from sensor_msgs.msg import Image as IMG
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber


import rospy
from scipy.spatial.transform import Rotation as RR,Slerp
from geometry_msgs.msg import PoseStamped
from dataset.pose_data import matrix_to_pose
from  policy.Utils import *

from std_msgs.msg import Float64MultiArray, Header

def sym_process(pose, sym='z'):
    x = pose[:3, 0]
    y = pose[:3, 1]
    z = pose[:3, 2]

    if sym == 'z':
        tmp_x = np.array([1, 0, 0])
        tmp_y = np.cross(z, tmp_x)
        tmp_y = tmp_y / np.linalg.norm(tmp_y)
        pose[:3, 0] = tmp_x
        pose[:3, 1] = tmp_y
        return pose

def process_rgbd(rgb, depth):
  rgb_tensor = torch.as_tensor(rgb[:, :480, :], device='cuda', dtype=torch.float)
  depth = torch.as_tensor(depth[:, :480], device='cuda', dtype=torch.float)
  depth = erode_depth(depth, radius=2, device='cuda')
  depth = bilateral_filter_depth(depth, radius=2, device='cuda')
  K = np.array([[455.15264892578125, 0, 327.128662109375],
                [0, 455.15264892578125, 240.3665771484375],
                [0, 0, 1]])
  xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[
    0]
  xyz_map = xyz_map[:, :480, :]
  xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)

  rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)
  xyz_map_tensor = xyz_map_tensor.permute(2, 0, 1).unsqueeze(0)

  rgb_tensor = F.interpolate(rgb_tensor, size=(160, 160), mode='bilinear', align_corners=False)
  xyz_map_tensor = F.interpolate(xyz_map_tensor, size=(160, 160), mode='bilinear', align_corners=False)

  A = torch.cat([rgb_tensor.cuda(), xyz_map_tensor.cuda()], dim=1).float()
  return A



def lerp(start, end, t):
    """Linear interpolation between start and end by a factor of t."""
    return (1 - t) * start + t * end


def slerp(start_rot, end_rot, t):
    """Spherical linear interpolation between two rotations by a factor of t."""
    # 创建旋转对象
    rotation_start = RR.from_euler('xyz',start_rot,degrees=False)
    rotation_end = RR.from_euler('xyz',end_rot,degrees=False)
    # 定义关键帧的时间点
    times = [0, 1]
    # 创建包含所有关键帧旋转的对象数组
    key_rots = RR.concatenate([rotation_start, rotation_end])
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



def transform_pose(source_in_camera, target_in_camera):
    T_camera_in_target = np.linalg.inv(target_in_camera)
    T_source_in_target = np.dot(T_camera_in_target, source_in_camera)

    pose_source_in_target = matrix_to_pose(T_source_in_target)
    return pose_source_in_target



def transform_poses_source(source_in_target, target_in_camera):
    """
    将 source_in_camera 和 target_in_camera 的位姿转换为 source_in_target 的位姿
    :param source_in_camera: 形状为 (345, 7) 的数组，表示 source 在相机坐标系下的位姿
    :param target_in_camera: 形状为 (345, 7) 的数组，表示 target 在相机坐标系下的位姿
    :return: 形状为 (345, 7) 的数组，表示 source 在 target 坐标系下的位姿
    """
    T_source_in_camera = np.dot(target_in_camera, source_in_target)

    return np.array(T_source_in_camera)



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
                    default='/media/agilex/HW/HUawei/PoseInsert/logs/502_img_pose_gate/policy_last.ckpt')
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



class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)

        self.arm_move()

        self.bridge = CvBridge()
        self.i = 1
        # 使用message_filters设置同步器
        self.rgb_sub = Subscriber("/camera_f/color/image_raw", IMG)
        self.depth_sub = Subscriber("/camera_f/depth/image_raw", IMG)
        self.astt = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1, slop=0.1)
        self.astt.registerCallback(self.image_callback)

        rospy.Subscriber("/target/pose", PoseStamped, self.target_pose_callback)
        rospy.Subscriber("/source/pose", PoseStamped, self.source_pose_callback)

        rospy.Subscriber("/target/bbox", Float64MultiArray, self.target_bbox_callback)
        rospy.Subscriber("/source/bbox", Float64MultiArray, self.source_bbox_callback)


        rospy.Subscriber("/right/pose_info", PoseStamped, self.arm_pose_callback)
        self.pub = rospy.Publisher("/right/pose_control", PoseStamped, queue_size=1000)



        self.normalize = True

        self.source_workspace = np.load('./source_workspace.npy')


    def target_bbox_callback(self, msg):
        self.target_bbox = np.array( msg.data)

    def source_bbox_callback(self, msg):
        self.source_bbox =  np.array( msg.data)

    def image_callback(self, rgb_msg, depth_msg):
        try:
            # 转换图像并放入队列以便后续处理
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgra8")
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1") # Y11  16UC1
        except CvBridgeError as e:
            print(e)

    def arm_pose_callback(self, msg):
        self.arm_pose = np.array( [msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z,
                                      msg.pose.orientation.x,
                                      msg.pose.orientation.y,
                                      msg.pose.orientation.z,
                                      msg.pose.orientation.w])

    def target_pose_callback(self, msg):
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

    def pose_to_matrix(self, pose):
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

    def process_images(self):

        args = deepcopy(default_args)
        for key, value in args_override.items():
            args[key] = value

        # set up device
        set_seed(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # policy
        print("Loading policy ...")
        policy =   PoseRGBD_DP(
            num_action=20,
            input_dim=9,
            obs_feature_dim=128,
            action_dim=9,
            hidden_dim=512,  
            # prediction="epsilon", ## epsilon
        ).to(device)
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

        # load checkpoint
        assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
        policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
        print("Checkpoint {} loaded.".format(args.ckpt))

        # ensemble_buffer = EnsembleBuffer(mode=args.ensemble_mode)

        cam_in_base = np.identity(4)
        cam_in_base[:3, :3] = RR.from_euler("xyz", np.array([-88.281, 2.101, -89.142]),
                                            degrees=True).as_matrix()
        cam_in_base[:3, 3] = np.array([-0.127, 0.292, -0.040])
        goal_colors = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/color_all.npy'.format(0))
        goal_depths = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/depth_all.npy'.format(0))
        source_bbox = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/source_bbox.npy'.format(0))
        target_bbox = np.load('/home/sunh/1RobotMPL/HUAWEI/train/{}/target_bbox.npy'.format(0))
        goal_color = goal_colors[len(goal_colors) - 1]
        goal_depth = goal_depths[len(goal_depths) - 1] /1000.0
        source_bbox = source_bbox[len(goal_colors)-1]
        target_bbox = target_bbox[len(goal_colors)-1]
        x_all = np.array([source_bbox[0], source_bbox[0] + source_bbox[2], target_bbox[0], target_bbox[0] + target_bbox[2]])
        y_all = np.array([source_bbox[1], source_bbox[1] + source_bbox[3], target_bbox[1], target_bbox[1] + target_bbox[3]])
        x_min, x_max = np.min(x_all), np.max(x_all)
        y_min, y_max = np.min(y_all), np.max(y_all)
        goal_color = goal_color[int(y_min):int(y_max), int(x_min):int(x_max)]
        goal_depth = goal_depth[int(y_min):int(y_max), int(x_min):int(x_max)]


        geta_all = []
        color_all = []
        while not rospy.is_shutdown():
            source_pose = self.source_pose
            target_pose = self.target_pose
            source_pose = self.pose_to_matrix(source_pose)
            target_pose = self.pose_to_matrix(target_pose)
            # source_pose = sym_process(source_pose)
            source_in_target = transform_pose(source_pose, target_pose)


            self.ee_in_base = self.arm_pose[:6]
            ee_in_base = np.identity(4)
            ee_in_base[:3, 3] = self.ee_in_base[:3]
            ee_in_base[:3, :3] = RR.from_euler("xyz", self.ee_in_base[3:6], degrees=False).as_matrix()
            source_in_base = np.dot(  cam_in_base  ,source_pose)
            ee_in_source = np.dot(np.linalg.inv(source_in_base),ee_in_base)



            if self.normalize:
                source_in_target = normalize_workspace(self.source_workspace, source_in_target)

            # 进行图像处理
            depth = np.asanyarray(self.depth_image)
            depth_full = np.zeros((480,640))
            depth_full[:400,:] = depth / 1000.0
            color = np.asanyarray(self.rgb_image)
            color = color[:, :, :3]
            color_show = color.copy()
            color_save = color.copy()

            source_bbox = self.source_bbox
            target_bbox = self.target_bbox
            x_all = np.array(
                [source_bbox[0], source_bbox[0] + source_bbox[2], target_bbox[0], target_bbox[0] + target_bbox[2]])
            y_all = np.array(
                [source_bbox[1], source_bbox[1] + source_bbox[3], target_bbox[1], target_bbox[1] + target_bbox[3]])
            x_min, x_max = np.min(x_all), np.max(x_all)
            y_min, y_max = np.min(y_all), np.max(y_all)
            color = color[int(y_min):int(y_max), int(x_min):int(x_max)]
            depth_full = depth_full[int(y_min):int(y_max), int(x_min):int(x_max)]


            rgbd = process_rgbd(color, depth_full)
            goal_rgbd = process_rgbd(goal_color, goal_depth)

            with torch.inference_mode():
                policy.eval()

                pose = np.array(source_in_target)
                pose = self.pose_to_matrix(pose)[:3, [0, 1, 3]]
                pose = torch.from_numpy(np.array([pose])).float().to(device)
                pose = pose.unsqueeze(0)

                # predict
                # actions = policy(rgbd, goal_rgbd, pose, actions=None, batch_size=1).squeeze(0).cpu().numpy()
                actions,gate_number = policy(rgbd, goal_rgbd, pose, actions=None, batch_size=1)
                actions = actions.squeeze(0).cpu().numpy()
                gate_number = gate_number.squeeze(0).cpu().numpy()
                geta_all.append(gate_number)
                color_all.append(color_save)
                source_in_target_action = actions[:, :9]


                if args.vis:
                    out_act_all = []
                    for i in range(5):
                        source_in_target = source_in_target_action[i]

                        source_in_target = source_in_target.reshape(3, 3)
                        T_source_in_target = np.identity(4)
                        x = source_in_target[:, 0]
                        y = source_in_target[:, 1]
                        z = np.cross(x, y)
                        T_source_in_target[:3, :2] = source_in_target[:3, :2]
                        T_source_in_target[:3, 2] = z
                        T_source_in_target[:3, 3] = source_in_target[:3, 2]
                        T_source_in_target = unnormalize_action(self.source_workspace, T_source_in_target)
                        T_source_in_cam = transform_poses_source(T_source_in_target, target_pose)

                        color_show = create_coor(color_show, source_pose[:3, :4], intrinsic_matrix,
                                                 corners_3D_size=0.02)
                        color_show = create_coor(color_show, target_pose[:3, :4], intrinsic_matrix,
                                                 corners_3D_size=0.02)
                        color_show = create_coor(color_show, T_source_in_cam[:3, :4], intrinsic_matrix,
                                                 corners_3D_size=0.05)
                        T_source_in_base = np.dot(cam_in_base, T_source_in_cam)
                        out_act = np.dot( T_source_in_base,ee_in_source)
                        out_act_all.append(out_act)

                    cv2.imshow('dp', color_show)
                    cv2.waitKey(0)

                    for i in range(5):
                        out_act = out_act_all[i]
                        out_eular = RR.from_matrix(out_act[:3, :3]).as_euler("xyz", degrees=False)
                        print("out_act: ",
                              [out_act[0, 3], out_act[1, 3], out_act[2, 3], out_eular[0], out_eular[1], out_eular[2]])

                        rate = rospy.Rate(30)
                        cv2.imshow('dp', color)
                        cv2.waitKey(0)
                        for irate in range(3):
                            pose_stamped = PoseStamped()
                            pose_stamped.header.stamp = rospy.Time.now()  # 设置时间戳为当前时间
                            pose_stamped.pose.position.x = out_act[0, 3]
                            pose_stamped.pose.position.y = out_act[1, 3]
                            pose_stamped.pose.position.z = out_act[2, 3]
                            pose_stamped.pose.orientation.x = out_eular[0]
                            pose_stamped.pose.orientation.y = out_eular[1]
                            pose_stamped.pose.orientation.z = out_eular[2]
                            pose_stamped.pose.orientation.w = 0.0
                            self.pub.publish(pose_stamped)
                            rate.sleep()
                np.save('geta_all.npy',np.array(geta_all))
                np.save('color_all.npy',np.array(color_all))






    def arm_move(self):
        print('arm_move')







if __name__ == '__main__':
    processor = ImageProcessor()
    try:
        processor.process_images()
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")







