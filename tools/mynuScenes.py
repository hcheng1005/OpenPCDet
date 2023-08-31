import argparse
import glob
from pathlib import Path

import open3d
from visual_utils import open3d_vis_utils as V

import os
import numpy as np
import argparse
import json
import sys
import numba
import yaml
import multiprocessing
import shutil
import cv2
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# 3DMOT module
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity


from nuscenes import NuScenes
import time


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
        points = np.fromfile(file_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
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
    data_path="/media/charles/ShareDisk/00myDataSet/nuScenes/v1.0-mini/sweeps/LIDAR_TOP/"
    cfg_file = "cfgs/nuscenes_models/cbgs_pp_multihead.yaml"
    ckpt = "ckpt/nuScenes/pp_multihead_nds5823_updated.pth"
    
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


def nuScenes_MOT(tracker, input_data: FrameData):
    results = tracker.frame_mot(input_data)
    return results

def frame_visualization(bboxes, ids, states, gt_bboxes=None, gt_ids=None, pc=None, dets=None, name=''):
    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12))
    if pc is not None:
        visualizer.handler_pc(pc)
    # for _, bbox in enumerate(gt_bboxes):
    #     visualizer.handler_box(bbox, message='', color='black')
    # dets = [d for d in dets if d.s >= 0.1]
    # for det in dets:
    #     visualizer.handler_box(det, message='%.2f' % det.s, color='green', linestyle='dashed')
    for _, (bbox, id, state) in enumerate(zip(bboxes, ids, states)):
        if Validity.valid(state):
            visualizer.handler_box(bbox, message=str(id), color='red')
        else:
            visualizer.handler_box(bbox, message=str(id), color='light_blue')
    visualizer.show()
    # visualizer.close()

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
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # 3D界面初始化
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # 跟踪器初始化
    # load model configs
    config_path = '../3DMOT_configs/nuScenes_cfg/giou.yaml'
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    tracker = MOTModel(configs)
    
    nusc = NuScenes(version='v1.0-mini',
                    dataroot='/media/charles/ShareDisk/00myDataSet/nuScenes/v1.0-mini/', 
                    verbose=True)

    with torch.no_grad():
        
        if 0:
            test_sersor = 'LIDAR_TOP'
            for scene_idx in range(len(nusc.scene)):
                scene = nusc.scene[scene_idx]

                # 获取该场景的first token
                cur_sample_info = nusc.get('sample', scene['first_sample_token'])
                # 连续读取该scenes下的所有frame
                while cur_sample_info['next'] != "":
                    path_ = nusc.get_sample_data_path(cur_sample_info['data'][test_sersor])
                    data_dict = demo_dataset.gen_data_dict(path_)
                    print(data_dict.keys())
                    # data_dict = demo_dataset.collate_batch([data_dict])
                    # print(data_dict.keys())
                    load_data_to_gpu(data_dict)
                    
                    pred_dicts, _ = model.forward(data_dict)
                    
                    V.draw_scenes(vis,
                            points=data_dict['points'][:, 1:],
                            ref_boxes=pred_dicts[0]['pred_boxes'],
                            ref_scores=pred_dicts[0]['pred_scores'],
                            ref_labels=pred_dicts[0]['pred_labels']
                            )
        else :
            for idx, data_dict in enumerate(demo_dataset):
                millis = int(round(time.time() * 1000))
                logger.info(f'Visualized sample index: \t{idx + 1}')
                 
                print(data_dict.keys())
                
                data_dict = demo_dataset.collate_batch([data_dict])
                # print(data_dict.keys())
                load_data_to_gpu(data_dict)
                
                pred_dicts, _ = model.forward(data_dict)

                # # CAMERA显示
                # image = cv2.imread(img_file_list[idx])
                # cv2.namedWindow("CAM FRONT", 0)
                # cv2.resizeWindow("CAM FRONT", 640, 480)
                # cv2.imshow('CAM FRONT', image)
                # cv2.waitKey(1)

                # 原始点云和检测显示
                V.draw_scenes(vis,
                            points=data_dict['points'][:, 1:],
                            ref_boxes=pred_dicts[0]['pred_boxes'],
                            ref_scores=pred_dicts[0]['pred_scores'],
                            ref_labels=pred_dicts[0]['pred_labels']
                            )

                # # 跟踪算法
                # ref_boxes = pred_dicts[0]['pred_boxes']
                # ref_boxes = ref_boxes.cpu().numpy()

                # ref_scores = pred_dicts[0]['pred_scores']
                # ref_scores = ref_scores.cpu().numpy()

                # ref_labels = pred_dicts[0]['pred_labels']
                # ref_labels = ref_labels.cpu().numpy()

                # # 格式转换
                # dets = np.hstack((ref_boxes, ref_scores.reshape([-1, 1])))

                # points = data_dict['points'][:, 1:]
                # points = points.cpu().numpy()

                # aux_info = {'is_key_frame': True}
                # dets = dets.tolist()

                # # print(dets.__len__())
                # new_dets=list()
                # for idx, det in enumerate(dets):
                #     if ref_labels[idx] == 1:
                #         new_dets.append(det)
                # print(new_dets.__len__())

                # # 构造MOT输入
                # frame_data = FrameData(dets=new_dets, ego=None, time_stamp=millis, pc=points, det_types=None, aux_info=aux_info)

                # #执行MOT算法
                # results = tracker.frame_mot(frame_data)
                # # print(results)
                # result_pred_bboxes = [trk[0] for trk in results]
                # result_pred_ids = [trk[1] for trk in results]
                # result_pred_states = [trk[2] for trk in results]
                # result_types = [trk[3] for trk in results]

                # visualization
                # if 1:
                #     frame_visualization(result_pred_bboxes, result_pred_ids, result_pred_states,
                #         gt_bboxes[frame_index], gt_ids[frame_index], frame_data.pc, dets=frame_data.dets, name='{:}_{:}_{:}'.format(args.name, sequence_id, frame_index))
                
                # frame_visualization(result_pred_bboxes, result_pred_ids, result_pred_states, name='xx')
                
                # # wrap for output
                # IDs.append(result_pred_ids)
                # result_pred_bboxes = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes]
                # bboxes.append(result_pred_bboxes)
                # states.append(result_pred_states)
                # types.append(result_types)

    # logger.info('nuScenes Demo done.')


if __name__ == '__main__':
    main()
