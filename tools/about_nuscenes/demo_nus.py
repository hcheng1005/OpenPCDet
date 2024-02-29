from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file
import torch
import numpy as np
import argparse
import glob
from pathlib import Path
import time

import open3d
from visual_utils import open3d_vis_utils as V
OPEN3D_FLAG = True


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


def parse_config():
    # data_path = "/media/charles/ShareDisk/00myDataSet/nuScenes/v1.0-mini/sweeps/LIDAR_TOP"
    # cfg_ = 'cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml'
    # ckpt_ = 'ckpt/nuScenes/cbgs_voxel01_centerpoint_nds_6454.pth'

    # transFusion
    data_path="/data/chenghao/mycode/private/OpenPCDet/data/nuscenes/v1.0-mini/sweeps/LIDAR_TOP"
    cfg_ = 'cfgs/nuscenes_models/transfusion_lidar.yaml'
    ckpt_ = 'ckpt/cbgs_transfusion_lidar.pth'
    
    # # # BEVFusion
    # data_path="/root/code/data/nuscenes/sweeps/LIDAR_TOP/"
    # cfg_ = 'cfgs/nuscenes_models/bevfusion.yaml'
    # ckpt_ = 'ckpt/cbgs_bevfusion.pth'
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_,
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=ckpt_,
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info(
        '-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    # 打印模型结构
    print(model)

    model.cuda()
    model.eval()

    # 3D界面初始化
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # print("pred_boxes \n", pred_dicts[0]['pred_boxes'])
            # print("pred_scores \n", pred_dicts[0]['pred_scores'])

            V.draw_scenes(vis,
                          points=data_dict['points'][:,1:], 
                          ref_boxes=pred_dicts[0]['pred_boxes'],
                          ref_scores=pred_dicts[0]['pred_scores'], 
                          ref_labels=pred_dicts[0]['pred_labels']
                          )

            time.sleep(0.1)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
