# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys; sys.path.insert(0, "./CenterPoint")

import os
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
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
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
    data_path="./testdata/kitti/000008.bin"
    cfg_ = '../output/cfgs/kitti_models/centerpoint_res/default/centerpoint_res.yaml'
    ckpt_ = '../output/cfgs/kitti_models/centerpoint_res/default/ckpt/checkpoint_epoch_40.pth'
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_,help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=data_path, help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=ckpt_, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


class CenterPointVoxelNet_Post(nn.Module):
    def __init__(self, model):
        super(CenterPointVoxelNet_Post, self).__init__()
        self.model = model
        
    def forward(self, x):
        
        # 此处可以认为是把模型裁剪了，只保留原生模型的neck和head部分
        # 因此需要注意，此处模型的输入x应该是原生模型backbone的输出（也就是3d稀疏卷积后的特征数据）
        x = self.model.backbone_2d(x)
        x = self.model.dense_head.shared_conv(x)
        
        # 多个检测任务（检测头）分离（此处应该是有6个）
        pred = [ task(x) for task in self.model.dense_head.heads_list]
        
        # return pred
        return pred[0]['center'], pred[0]['center_z'], pred[0]['dim'], pred[0]['rot'], pred[0]['hm']
    


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

    # model.eval()
    # post_model.eval()

    # model = model.cuda()

    if(1): 
        post_model = post_model.cuda()
        post_model.half()
    
        rpn_input = torch.load("spatial_features.tensor")
        rpn_input = rpn_input.half()
        # onnx_output =  post_model.forward(rpn_input)
        # torch.save(onnx_output, "onnx_output.tensor")
        
        # 调用onnx-export进行模型导出
        # rpn_input = data_dict['spatial_features'], 
        torch.onnx.export(post_model, rpn_input, "tmp.onnx",
            export_params=True, opset_version=11, do_constant_folding=True,
            keep_initializers_as_inputs=False, input_names = ['input'],
            output_names = ['reg_0', 'height_0', 'dim_0', 'rot_0', 'hm_0'],
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