import argparse
import glob
from pathlib import Path
from xml.etree.ElementTree import PI
import math
from pandas import read_pickle
import poyo

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset

import matplotlib.pyplot as plt

import open3d
from visual_utils import open3d_vis_utils as V

import numpy as np
import torch

import cv2

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

#3DMOT module
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData


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
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        points = np.fromfile(self.sample_file_list[index], dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

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
    cfg_file="cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml"
    ckpt="cfgs/ckpt/nuScenes/cbgs_voxel01_centerpoint_nds_6454.pth"
    data_path="/media/charles/ShareDisk/00myDataSet/nuScenes/00SourceFile/v1.0-trainval/samples/LIDAR_TOP/"
    # data_path='/home/charles/myDataSet/nuScenes/v1.0-mini/samples/LIDAR_TOP/'

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_file,
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=ckpt, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick nuScenes Demo of OpenPCDet-------------------------')

    #加载数据集
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, 
        class_names=cfg.CLASS_NAMES, 
        training=False,
        root_path=Path(args.data_path),  logger=logger
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    #加载模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    
    #导入模型数据
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    #3D界面初始化
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # plt.figure()
    # plt.ion() 
    # plt.axis('equal')
    img_path = "/media/charles/ShareDisk/00myDataSet/nuScenes/00SourceFile/v1.0-trainval/samples/CAM_FRONT/"
    img_file_list = glob.glob(str(Path(img_path) / f'*.jpg'))
    img_file_list.sort()

    # print(img_file_list[0])
    # image = cv2.imread(img_file_list[0])
    # cv2.imshow('xx',image)
    # cv2.waitKey(0)

    # return
    

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            #CAMERA显示
            image = cv2.imread(img_file_list[idx])
            cv2.namedWindow("CAM FRONT",0)
            cv2.resizeWindow("CAM FRONT", 640, 480)
            cv2.imshow('CAM FRONT',image)
            cv2.waitKey(1)
            
            #原始点云和检测显示
            V.draw_scenes(vis,
                points=data_dict['points'][:, 1:], 
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], 
                ref_labels=pred_dicts[0]['pred_labels']
            )

            # 跟踪算法 

            # points=data_dict['points'][:, 1:]
            # pred_boxes=pred_dicts[0]['pred_boxes']

            # if isinstance(points, torch.Tensor):
            #     points = points.cpu().numpy()

            # if isinstance(pred_boxes, torch.Tensor):
            #     pred_boxes = pred_boxes.cpu().numpy()

            # pred_boxes_array=np.array(points) 
            # ref_boxes_array = (np.array(pred_boxes))
 
            # #绘制原始点云数据
            # plt.scatter(pred_boxes_array[:, 0], pred_boxes_array[:, 1], marker='o', s=0.01)

            # #绘制检测dets
            # for i in range(pred_boxes.shape[0]):
            #     # if ref_labels[i] < 2:
            #         # line_set, box3d = translate_boxes_to_open3d_instance(pred_boxes[i])
            #     pred_boxes2 = pred_boxes[i]
            #     center = pred_boxes2[0:3]
            #     lwh = pred_boxes2[3:6]
            #     axis_angles = np.array([0, 0, pred_boxes2[6] + 1e-10])
            #     rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
            #     box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
            #     corners2 = np.asarray(box3d.get_box_points())
            #     # print(corners2)
            #     corners = np.array(corners2[:, :2])
            #     corners = np.concatenate([corners, corners[0:1, :2]])
            #     plt.plot(corners[:, 0], corners[:, 1], linestyle='solid')

            # plt.show()
            # plt.pause(0.1)
            # plt.clf()
            # # for i in range(np.shape(ref_boxes)[0]):
            #     # print(i)
                

    logger.info('nuScenes Demo done.')


if __name__ == '__main__':
    main()
