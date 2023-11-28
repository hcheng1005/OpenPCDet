
import sys; sys.path.insert(0, "./CenterPoint")

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pickle
import torch
import onnx
import argparse
from onnxsim import simplify
import numpy as np
from torch import nn
from pathlib import Path

import glob

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate


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
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def simplify_model(model_path):
    model = onnx.load(model_path)
    if model is None:
        print("File %s is not find! "%model_path)
    return simplify(model)

def parse_config():    
    data_path="/data/wz/0.bin"
    cfg_ = "ckpt/wuzhneg/model_ct_1018/ownwz_centerpoint_10cls_1018.yaml"
    ckpt_ = "ckpt/wuzhneg/model_ct_1018/checkpoint_epoch_80.pth"
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_,help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=data_path, help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=ckpt_, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


# For WuZheng
class CenterPointVoxelNet_Post(nn.Module):
    def __init__(self, model):
        super(CenterPointVoxelNet_Post, self).__init__()
        self.model = model
        assert( len(self.model.dense_head.heads_list) == 6 )

    def forward(self, x):
        x = self.model.backbone_2d(x)
        x = self.model.dense_head.shared_conv(x)
        
        # 多个检测任务（检测头）分离（此处应该是有6个）
        pred = [ task(x) for task in self.model.dense_head.heads_list ]

        return pred[0]['center'], pred[0]['center_z'], pred[0]['dim'], pred[0]['rot'],  pred[0]['hm'], \
                pred[1]['center'], pred[1]['center_z'], pred[1]['dim'], pred[1]['rot'],  pred[1]['hm'], \
                pred[2]['center'], pred[2]['center_z'], pred[2]['dim'], pred[2]['rot'],  pred[2]['hm'], \
                pred[3]['center'], pred[3]['center_z'], pred[3]['dim'], pred[3]['rot'], pred[3]['hm'], \
                pred[4]['center'], pred[4]['center_z'], pred[4]['dim'], pred[4]['rot'], pred[4]['hm'], \
                pred[5]['center'], pred[5]['center_z'], pred[5]['dim'], pred[5]['rot'],  pred[5]['hm']
                
def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    # 构造原生模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    
    # print(model)
    
    # 截取模型的neck和head部分
    post_model = CenterPointVoxelNet_Post(model)

    model.eval()
    post_model.eval()

    model = model.cuda()

    if(1): 
        post_model = post_model.cuda()
        post_model.half()
        
        # spatial_features size = [1, 256, 64, 128]
        rpn_input =torch.zeros((1, 256, 64, 128), dtype=torch.float16, device='cuda:0') # WUZHENG
        
        torch.onnx.export(post_model, rpn_input, "tmp.onnx",
            export_params=True, opset_version=11, do_constant_folding=True,
            keep_initializers_as_inputs=False, input_names = ['input'],
            output_names = ['reg_0', 'height_0', 'dim_0', 'rot_0', 'hm_0',
                            'reg_1', 'height_1', 'dim_1', 'rot_1', 'hm_1',
                            'reg_2', 'height_2', 'dim_2', 'rot_2', 'hm_2',
                            'reg_3', 'height_3', 'dim_3', 'rot_3', 'hm_3',
                            'reg_4', 'height_4', 'dim_4', 'rot_4', 'hm_4',
                            'reg_5', 'height_5', 'dim_5', 'rot_5', 'hm_5'],
            )

        # 模型简化和保存最终模型onnx
        sim_model, check = simplify_model("tmp.onnx")
        if not check:
            print("[ERROR]:Simplify %s error!"% "tmp.onnx")
        onnx.save(sim_model, "centerpoint_rpn.onnx")
        print("[PASS] Export ONNX done.")
       
    if(0):        
        with torch.no_grad():
            for idx, data_dict in enumerate(demo_dataset):
                logger.info(f'Visualized sample index: \t{idx + 1}')
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)

                model.forward(data_dict)
                            
                torch.save(data_dict['spatial_features'], "spatial_features_10.tensor")
                torch.save(data_dict['pred_dicts'],"troch_box_dicts_10.tensor")


if __name__ == '__main__':
    main()