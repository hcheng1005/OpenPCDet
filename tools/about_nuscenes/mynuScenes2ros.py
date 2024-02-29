import argparse
import glob
from pathlib import Path

import os
import numpy as np
import argparse

import cv2
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from nuscenes import NuScenes
import time
import open3d
from visual_utils import open3d_vis_utils as V

# ros
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from scipy.spatial.transform import Rotation as R

import numpy as np
from std_msgs.msg import Header


import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default="yolox-m", help="model name")

    parser.add_argument(
        "--path", default="/media/charles/ShareDisk/00myDataSet/nuScenes/v1.0-mini/sweeps/CAM_FRONT", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default="/home/charles/myCode/about_camera/YOLOX/pth/yolox_m.pth", type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="gpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


class Detector:
    def __init__(self) -> None:
        self.marker_pub = rospy.Publisher(
            '/detect_box3d', MarkerArray, queue_size=2)

        self.marker_array = MarkerArray()

    def rotx(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1,  0,  0],
                        [0,  c,  -s],
                        [0, s,  c]])

    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                        [0,  1,  0],
                        [-s, 0,  c]])

    def rotz(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  -s,  0],
                        [s,  c,  0],
                        [0, 0,  1]])

    def get_3d_box(self, center, box_size, heading_angle):
        ''' Calculate 3D bounding box corners from its parameterization.

        Input:heading_angle
            box_size: tuple of (l,w,h)
            : rad scalar, clockwise from pos z axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''
        R = self.rotz(heading_angle)
        l, w, h = box_size
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[0]
        corners_3d[1, :] = corners_3d[1, :] + center[1]
        corners_3d[2, :] = corners_3d[2, :] + center[2]
        corners_3d = np.transpose(corners_3d)
        return corners_3d

    def display(self, boxes):
        self.marker_array.markers.clear()

        for obid in range(len(boxes)):
            ob = boxes[obid]
            tid = 0
            detect_points_set = []
            for i in range(0, 8):
                detect_points_set.append(Point(ob[i], ob[i+8], ob[i+16]))

            marker = Marker()
            marker.header.frame_id = 'nuscenes'
            marker.header.stamp = rospy.Time.now()

            marker.id = obid*2
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST

            marker.lifetime = rospy.Duration(1)

            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0

            marker.color.a = 1
            marker.scale.x = 0.2
            marker.points = []

            for line in lines:
                marker.points.append(detect_points_set[line[0]])
                marker.points.append(detect_points_set[line[1]])

        self.marker_array.markers.append(marker)

        self.marker_pub.publish(self.marker_array)


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(
            str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        points = np.fromfile(
            self.sample_file_list[index], dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)

        input_dict = {
            'points': points,
            'frame_id': times,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def gen_data_dict(self, file_path):
        points = np.fromfile(file_path, dtype=np.float32,
                             count=-1).reshape([-1, 5])[:, :4]
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)

        input_dict = {
            'points': points,
            'frame_id': times,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    data_path = "/media/charles/ShareDisk/00myDataSet/nuScenes/v1.0-mini/sweeps/LIDAR_TOP/"
    cfg_file = "cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml"
    ckpt = "ckpt/nuScenes/cbgs_voxel01_centerpoint_nds_6454.pth"

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_file,
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=ckpt,
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def pub_3d_bbox2(box_pub, pred_dicts):
    detector = Detector()
    if isinstance(pred_dicts, torch.Tensor):
        pred_dicts = pred_dicts.cpu().numpy()

    print("det numb is ", pred_dicts.shape[0])
    boxes = []
    for i in range(pred_dicts.shape[0]):
        box = detector.get_3d_box(
            pred_dicts[i][0:3], pred_dicts[i][3:6], pred_dicts[i][6])
        box = box.transpose(1, 0).ravel()
        boxes.append(box)

    detector.display(boxes)


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return lines, box3d


def pub_3d_bbox(box_pub, pred_dicts, pred_labels, pred_score):
    bboxes = MarkerArray()

    if isinstance(pred_dicts, torch.Tensor):
        pred_dicts = pred_dicts.cpu().numpy()

    pred_score = pred_score.cpu().numpy()
    print("det numb is ", pred_dicts.shape[0])
    for i in range(pred_dicts.shape[0]):
        # line_set, box3d = translate_boxes_to_open3d_instance(pred_dicts[i])

        if pred_labels[i] < 3 and pred_score[i] > 0.2:

            marker = Marker()
            marker.id = i
            marker.header.frame_id = "nuscenes"  # 设置坐标系
            marker.type = Marker.CUBE  # 指定标记类型为立方体
            marker.action = Marker.ADD
            marker.scale.x = pred_dicts[i][3]  # 立方体的尺寸
            marker.scale.y = pred_dicts[i][4]
            marker.scale.z = pred_dicts[i][5]
            marker.color.a = 0.3  # 不透明
            marker.color.r = 0.0  # 红色
            marker.color.g = 1.0  # 红色
            marker.color.b = 0.0  # 红色

            r = R.from_euler('xyz', [0, 0, pred_dicts[i][6]])

            marker.pose.orientation.x = r.as_quat()[0]
            marker.pose.orientation.y = r.as_quat()[1]
            marker.pose.orientation.z = r.as_quat()[2]
            marker.pose.orientation.w = r.as_quat()[3]

            marker.lifetime = rospy.Duration(1.0)  # 显示时间

            # 设置立方体的位置
            marker.pose.position.x = pred_dicts[i][0]
            marker.pose.position.y = pred_dicts[i][1]
            marker.pose.position.z = pred_dicts[i][2]
            
            # print(pred_score[i])
            marker.text = str(pred_score[i])
            
            # print(pred_score[i], marker.text )

            bboxes.markers.append(marker)

    box_pub.publish(bboxes)

def YOLOX_main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.cuda()
    model.eval()

    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    predictor = Predictor(model, exp, COCO_CLASSES)
    return predictor
    # if os.path.isdir(args.path):
    #     files = get_image_list(args.path)
    # else:
    #     files = [args.path]
    # files.sort()
    # for image_name in files:
    #     outputs, img_info = predictor.inference(image_name)
    #     result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    #     cv2.imshow('demo', result_image)
    #     cv2.waitKey(20)
        
def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info(
        '-----------------Quick nuScenes Demo of OpenPCDet-------------------------')

    # 加载数据集
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),  logger=logger
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # 加载摄像头数据
    img_path = "/media/charles/ShareDisk/00myDataSet/nuScenes/v1.0-trainval/sweeps/CAM_FRONT/"
    img_file_list = glob.glob(str(Path(img_path) / f'*.jpg'))
    img_file_list.sort()

    # 加载模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=demo_dataset)

    # 导入模型数据
    model.load_params_from_file(
        filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    nusc = NuScenes(version='v1.0-mini',
                    dataroot='/media/charles/ShareDisk/00myDataSet/nuScenes/v1.0-mini/',
                    verbose=True)

    pub = rospy.Publisher('pointcloud_topic', PointCloud2, queue_size=5)
    rospy.init_node('pointcloud_publisher_node', anonymous=True)
    rate = rospy.Rate(1)
    msg = PointCloud2()
    msg.header.stamp = rospy.Time().now()
    msg.header.frame_id = "nuscenes"

    ros_img = Image()
    pub_img = rospy.Publisher("/front_camera/image_raw", Image, queue_size=2)

    box_pub = rospy.Publisher('visualization_marker',
                              MarkerArray, queue_size=10)
    
    # 加载摄像头模型
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    yolox_predictor = YOLOX_main(exp, args)
    
    with torch.no_grad():
        test_sersor = 'LIDAR_TOP'
        for scene_idx in range(len(nusc.scene)):
            scene = nusc.scene[scene_idx]

            # 获取该场景的first token
            cur_sample_info = nusc.get('sample', scene['first_sample_token'])
            print(cur_sample_info)

            # 连续读取该scenes下的所有frame
            while cur_sample_info['next'] != "":
                print(cur_sample_info['next'])
                
                path_ = nusc.get_sample_data_path(
                    cur_sample_info['data'][test_sersor])
                # print(path_)
                data_dict = demo_dataset.gen_data_dict(path_)
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)

                pred_dicts, _ = model.forward(data_dict)

                # 读取camera
                img_path_ = nusc.get_sample_data_path(
                    cur_sample_info['data']['CAM_FRONT'])
                
                outputs, img_info = yolox_predictor.inference(img_path_)
                result_image = yolox_predictor.visual(outputs[0], img_info, yolox_predictor.confthre)
                # image = cv2.imread(img_path_)
                header = Header(stamp=rospy.Time.now())
                header.frame_id = 'nuscenes'
                ros_img.encoding = 'bgr8'
                ros_img.header = header
                ros_img.height = result_image.shape[0]
                ros_img.width = result_image.shape[1]
                ros_img.step = result_image.shape[1] * result_image.shape[2]
                ros_img.data = np.array(result_image).tostring()

                # 读取lidar数据
                points = data_dict['points'][:, 1:]
                if isinstance(points, torch.Tensor):
                    points = points.cpu().numpy()
                points = points[:, :3]
                msg.height = 1
                msg.width = len(points)
                print(len(points))
                msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                              PointField('y', 4, PointField.FLOAT32, 1),
                              PointField('z', 8, PointField.FLOAT32, 1)]
                msg.is_bigendian = False
                msg.point_step = 12
                msg.row_step = msg.point_step * points.shape[0]
                msg.is_dense = False
                msg.data = np.asarray(points, np.float32).tostring()

                # 发布lidar
                pub.publish(msg)
                # 发布camera
                pub_img.publish(ros_img)

                pub_3d_bbox(
                    box_pub, pred_dicts[0]['pred_boxes'], pred_dicts[0]['pred_labels'], pred_dicts[0]['pred_scores'])

                rate.sleep()

                # next
                cur_sample_info = nusc.get('sample', cur_sample_info['next'])


if __name__ == '__main__':
    main()
