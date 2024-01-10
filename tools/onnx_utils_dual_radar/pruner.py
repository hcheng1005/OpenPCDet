
import os
import argparse
import re
import glob
import onnx
import argparse
import numpy as np
import datetime
from pathlib import Path
from onnxsim import simplify

import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file

import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from exporter_paramters import export_paramters as export_paramters
from simplifier_onnx import simplify_preprocess, simplify_postprocess

# import torch_pruning as tp
import pruning.torch_pruning_v0 as tp

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
            self.sample_file_list[index], dtype=np.float32, count=-1).reshape([-1, 5])
        
        print(points.shape)

        sweep_points_list = [points]

        points = np.concatenate(sweep_points_list, axis=0)

        input_dict = {
            'points': points,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
    #                             pre_trained_path=args.pretrained_model)
    # model.cuda()
    
    # start evaluation
    eval_utils.eval_one_epoch(cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )
    
def parse_config():
    data_path="/home/charles/code/OpenPCDet/data/dual_radar/radar_arbe/testing/arbe"
    cfg_ = "cfgs/dual_radar_models/pointpillar_arbe.yaml"
    ckpt_ = "ckpt/dual_radar/pointpillars_arbe_80.pth"

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_,
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=ckpt_,
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    # parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


if __name__ == "__main__":
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )   
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model = model.cuda()
    model = model.eval()
    
    prune_ratio = 0.2
    granularity = 8

    unpruned_total_params = sum(p.numel() for p in model.parameters())
    
    # 1. build dependency graph
    strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()
    DG = tp.DependencyGraph()
    
    input_dict = next(iter(demo_dataset))
    print(input_dict.keys())
    input_dict = demo_dataset.collate_batch([input_dict])
    load_data_to_gpu(input_dict)
    DG.build_dependency(model, example_inputs=input_dict)

    # conv layers
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    # print(layers)
    # Exclude heads
    black_list = layers[-3:]
    # print(black_list)
    
    count = 0
    for layer in layers:
        if layer in black_list:
            continue
        # can run some algo here to generate threshold for every node
        threshold_run = 0.01
        pruning_idxs = strategy(layer.weight, amount=threshold_run)
        pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=pruning_idxs)
        if pruning_plan is not None:
            pruning_plan.exec()
        else:
            continue
        count += 1
        
    pruned_total_params = sum(p.numel() for p in model.parameters())
    print("Pruning ratio: {}".format(pruned_total_params / unpruned_total_params))
    
    import pcdet
    version = 'pcdet+' + pcdet.__version__
    stat_ = {'epoch': 1, 'it': None, 'model_state': model.state_dict(), 'optimizer_state': None, 'version': version}
    torch.save(stat_, './pruned_pp.pth', _use_new_zipfile_serialization=False)
    
    # print(model)
    
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=args.workers, logger=logger, training=False
    )
    
    eval_output_dir =Path('./')
    eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, 1)



    