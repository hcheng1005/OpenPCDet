import argparse
import glob
from pathlib import Path

import open3d
from visual_utils import open3d_vis_utils as V

import os, numpy as np, argparse, json, sys, numba, yaml, multiprocessing, shutil, cv2
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

#3DMOT module
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData

import cv2

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
    data_path="/home/charles/myDataSet/nuScenes/v1.0-mini/samples/LIDAR_TOP/"
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


def nuScenes_MOT(tracker, input_data: FrameData):
    results = tracker.frame_mot(input_data)
    return results


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

    # 加载摄像头数据
    img_path = "/home/charles/myDataSet/nuScenes/v1.0-mini/samples/CAM_FRONT/"
    img_file_list = glob.glob(str(Path(img_path) / f'*.jpg'))
    img_file_list.sort()

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

    # 跟踪器初始化
    # load model configs
    config_path ='../3DMOT_configs/nuScenes_cfg/giou.yaml'
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    tracker = MOTModel(configs)
    IDs, bboxes, states, types = list(), list(), list(), list()

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # # CAMERA显示
            # image = cv2.imread(img_file_list[idx])
            # cv2.namedWindow("CAM FRONT",0)
            # cv2.resizeWindow("CAM FRONT", 640, 480)
            # cv2.imshow('CAM FRONT',image)
            # cv2.waitKey(1)
            
            # 原始点云和检测显示
            V.draw_scenes(vis,
                points=data_dict['points'][:, 1:], 
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], 
                ref_labels=pred_dicts[0]['pred_labels']
            )

            # 跟踪算法 
            ref_boxes=pred_dicts[0]['pred_boxes']
            if isinstance(ref_boxes, torch.Tensor):
                ref_boxes = ref_boxes.cpu().numpy()
            frame_data = FrameData(dets=ref_boxes, ego=None, time_stamp=None, pc=None, det_types=None, aux_info=None)
            results = tracker.frame_mot(frame_data)
            print(results)

                
    logger.info('nuScenes Demo done.')


if __name__ == '__main__':
    main()
