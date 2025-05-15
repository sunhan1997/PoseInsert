#!/usr/bin/env python
import time
import numpy as np
from sensor_msgs.msg import Image as IMG
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

from estimater import *
from datareader import *
import argparse
# from ultralytics import YOLO
import rospy
from std_msgs.msg import Float64MultiArray, Header
from scipy.spatial.transform import Rotation as RR
import tf2_ros

from geometry_msgs.msg import PoseStamped

from plyfile import PlyData


def create_bounding_box(img, pose, pt_cld_data, intrinsic_matrix,color=(0,0,255)):
    "Create a bounding box around the object"
    # 8 corner points of the ptcld data
    min_x, min_y, min_z = np.min(pt_cld_data['x']), np.min(pt_cld_data['y']),np.min(pt_cld_data['z'])
    max_x, max_y, max_z = np.max(pt_cld_data['x']), np.max(pt_cld_data['y']),np.max(pt_cld_data['z'])

    corners_3D = np.array([[max_x, min_y, min_z],
                           [max_x, min_y, max_z],
                           [min_x, min_y, max_z],
                           [min_x, min_y, min_z],
                           [max_x, max_y, min_z],
                           [max_x, max_y, max_z],
                           [min_x, max_y, max_z],
                           [min_x, max_y, min_z]])

    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)

    # Perspective Projection to obtain 2D coordinates for masks
    homogenous_2D = intrinsic_matrix @ (pose @ homogenous_coordinate.T)
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)

    # Draw lines between these 8 points
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), color, 1)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), color, 1)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[4]), color, 1)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[2]), color, 1)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[5]), color, 1)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[3]), color, 1)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[6]), color, 1)
    img = cv2.line(img, tuple(coord_2D[3]), tuple(coord_2D[7]), color, 1)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[7]), color, 1)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[5]), color, 1)
    img = cv2.line(img, tuple(coord_2D[5]), tuple(coord_2D[6]), color, 1)
    img = cv2.line(img, tuple(coord_2D[6]), tuple(coord_2D[7]), color, 1)

    return img


def create_coor(img, pose, intrinsic_matrix):
    "Create a bounding box around the object"
    # 8 corner points of the ptcld data

    x_color = (255, 0, 0)
    y_color = (0, 255, 0)
    z_color = (0, 0, 255)
    corners_3D = np.array([[0, 0, 0],
                           [0.05, 0, 0],
                           [0, 0.05, 0],
                           [0, 0, 0.05]])

    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)

    # Perspective Projection to obtain 2D coordinates for masks
    homogenous_2D = intrinsic_matrix @ (pose @ homogenous_coordinate.T)
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)
    # coord_2D = ((np.floor(coord_2D))).astype(int)
    # Draw lines between these 8 points
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), x_color, 3)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[2]), y_color, 3)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), z_color, 3)

    return img

def calc_2d_bbox(xs, ys, im_size):
    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))
    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]

class BoundingBoxDrawer:
    def __init__(self, color):
        self.image = color

        self.image_copy = self.image.copy()
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1
        self.window_name = 'Draw Bounding Box'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.draw_rectangle)

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.image = self.image_copy.copy()
                cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fx, self.fy = x, y
            cv2.rectangle(self.image, (self.ix, self.iy), (self.fx, self.fy), (0, 255, 0), 2)

    def run(self):
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按'q'键退出
                self.save_mask()
                break
            # elif key == ord('s') and not self.drawing:  # 按's'键保存mask
            #     self.save_mask()


        cv2.destroyAllWindows()

    def save_mask(self):
        if self.ix != -1 and self.iy != -1 and self.fx != -1 and self.fy != -1:
            self.mask = np.zeros_like(self.image)
            self.mask = cv2.rectangle(self.mask, (self.ix, self.iy), (self.fx, self.fy), (255, 255, 255), -1)
            cv2.imwrite('mask.png', self.mask)
            print("Mask已保存为'mask.png'。")

obj_DIR = 'e_base' ## 1: base_small 2: base_big  3:base base_2  Screw_Base 502_base  base_1   e_base
id_obj = 1
model_path2 = os.path.join('./demo_data/{}/mesh/{}.ply'.format(obj_DIR ,obj_DIR))
ply2 = PlyData.read(model_path2)
pt_cld_data = ply2.elements[0].data

parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/{obj_DIR}/mesh/{obj_DIR}.ply')
# parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/my/{obj_DIR}/mesh/GS185-Li-reconstruct.ply')
parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/{obj_DIR}')
parser.add_argument('--est_refine_iter', type=int, default=5)
parser.add_argument('--track_refine_iter', type=int, default=2)
parser.add_argument('--debug', type=int, default=3)
parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
args = parser.parse_args()

set_logging_format()
set_seed(0)

mesh = trimesh.load(args.mesh_file)

debug = args.debug
debug_dir = args.debug_dir
os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                     refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
logging.info("estimator initialization done")

reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)


class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)
        self.bridge = CvBridge()
        self.i = 1
        # 使用message_filters设置同步器
        self.rgb_sub = Subscriber("/camera_f/color/image_raw", IMG) # camera_f  camera_l
        self.depth_sub = Subscriber("/camera_f/depth/image_raw", IMG)
        # self.result_pub = rospy.Publisher("/camera_f/hand_pose_image", IMG, queue_size=1)

        self.astt = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1, slop=0.1)
        self.astt.registerCallback(self.image_callback)
        self.pub = rospy.Publisher("/target/pose", PoseStamped, queue_size=1000)

        self.pub_bbox = rospy.Publisher("/target/bbox", Float64MultiArray, queue_size=1000)



    def image_callback(self, rgb_msg, depth_msg):
        try:
            # 转换图像并放入队列以便后续处理
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgra8")
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1") # Y11  16UC1
        except CvBridgeError as e:
            print(e)



    def process_images(self):
        while not rospy.is_shutdown():
            # 进行图像处理
            depth = np.asanyarray(self.depth_image)
            print('************************************************')
            print(depth.shape)
            print('************************************************')
            color = np.asanyarray(self.rgb_image)
            color = color[:, :, :3]
            # color = cv2.convertScaleAbs(color, alpha=1.0, beta=-50.) ### 降低亮度
            depth_full = np.zeros((480,640))
            depth_full[:400,:] = depth

            if self.i == 1:
                cv2.imwrite('zed_ros_image.png', color)
                np.save('depth_image', depth_full)

                #######################################################3
                drawer = BoundingBoxDrawer(color)
                drawer.run()
                mask = drawer.mask


                color = reader.get_color_real(color)
                depth = reader.get_depth_real(depth_full)
                mask = reader.get_mask_real(mask).astype(bool)
                pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

                print(pose)

                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                                    is_input_rgb=True)

                # result_image = self.bridge.cv2_to_imgmsg(vis, "bgr8")
                # self.result_pub.publish(result_image)
                cv2.imshow('target', vis)
                cv2.waitKey(0)
                self.i = 2
            else:
                color = reader.get_color_real(color)
                depth = reader.get_depth_real(depth_full)
                pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
                print(pose)

                center_pose = pose #@ np.linalg.inv(to_origin)
                # vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                # vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                #                     is_input_rgb=True)
                vis = create_coor(color, center_pose[:3,:4],  intrinsic_matrix=reader.K)
                rgb_show = np.zeros((480, 640))
                img_show = create_bounding_box(rgb_show, pose[:3, :4], pt_cld_data, reader.K, color=(255, 255, 255))  # red
                ys, xs = np.nonzero(img_show > 0)
                tar_bbox = calc_2d_bbox(xs, ys, (640, 480))  # linemod
                x1, y1, w1, h1 = tar_bbox

                cv2.rectangle(vis, (x1, y1), ((x1+w1), (y1+h1)), (0, 255, 0), 2)
                cv2.imshow('target', vis)
                cv2.waitKey(1)
                # result_image = self.bridge.cv2_to_imgmsg(vis, "bgr8")
                # self.result_pub.publish(result_image)

                r = RR.from_matrix(pose[0:3, 0:3]) #center_pose
                qua = r.as_quat()

                detect_result = Float64MultiArray()
                detect_result.data = [x1, y1, w1, h1]
                self.pub_bbox .publish(detect_result)

                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = rospy.Time.now()  # 设置时间戳为当前时间
                pose_stamped.pose.position.x = pose[0, 3]
                pose_stamped.pose.position.y = pose[1, 3]
                pose_stamped.pose.position.z = pose[2, 3]
                pose_stamped.pose.orientation.x = qua[0]
                pose_stamped.pose.orientation.y = qua[1]
                pose_stamped.pose.orientation.z = qua[2]
                pose_stamped.pose.orientation.w = qua[3]
                self.pub.publish(pose_stamped)



    def spin(self):
        time.sleep(1)
        rate = rospy.Rate(30)


        while not rospy.is_shutdown():
            if hasattr(self, "rgb_image") and hasattr(self, "depth_image"):
                self.process_images()
            rate.sleep()


if __name__ == '__main__':
    processor = ImageProcessor()
    try:
        processor.spin()
    except KeyboardInterrupt:
        print("shutting down")
    cv2.destroyAllWindows()


#########################################################  link 6 in camera_f ##############################################3

# [[-0.9082755  -0.19185421 -0.37178978  0.06266354]
#  [ 0.18225054 -0.9813476   0.06116892 -0.05076155]
#  [-0.37659052 -0.01220062  0.9262995   0.3479252 ]
#  [ 0.          0.          0.          1.        ]]


# [[-0.9744784  -0.17410675 -0.14169817  0.06671117]
#  [ 0.17062934 -0.9846614   0.03642736 -0.05016922]
#  [-0.14586715  0.0113198   0.98923904  0.33627304]
#  [ 0.          0.          0.          1.        ]]


# [[-0.9745228  -0.182744   -0.13003704  0.0661073 ]
#  [ 0.17971264 -0.983104    0.03477753 -0.05080495]
#  [-0.13419554  0.01052212  0.9908989   0.33653763]
#  [ 0.          0.          0.          1.        ]]

#########################################################  link 6 in camera_f ##############################################3

  # position:
  #   x: 0.047523438930511475
  #   y: 0.3487545847892761
  #   z: -0.009661011397838593
  # orientation:
  #   x: 1.5056909322738647
  #   y: 0.1565750390291214
  #   z: 1.800633430480957
  #   w: -0.00286102294921875


#########################################################  link 6 in camera_f ##############################################3