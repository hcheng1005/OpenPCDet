# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import onnx
import torch
import argparse
import numpy as np
import onnxruntime

from pathlib import Path
from onnxsim import simplify
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file

from pcdet.models.backbones_3d import VoxelBackBone8x

# from exporter_paramters import export_paramters as export_paramters
from simplifier_nus_multpp_onnx import simplify_preprocess, simplify_postprocess

import math

from torch.onnx.symbolic_registry import register_op 
 

def atan2_symbolic(g, self, other):
    # self is y, and other is x on coordinate
    slope = g.op("Div", self, other)
    atan = g.op("Atan", slope)
    const_zero = g.op("Constant", value_t=torch.tensor(0).to('cuda:0'))
    const_pi = g.op("Constant", value_t=torch.tensor(math.pi).to('cuda:0'))

    condition_second_or_third_quadrant = g.op("Greater", self, const_zero)
    second_third_quadrant = g.op(
        "Where",
        condition_second_or_third_quadrant,
        g.op("Add", atan, const_pi),
        g.op("Sub", atan, const_pi),
    )

    condition_14_or_23_quadrant = g.op("Less", other, const_zero)
    result = g.op("Where", condition_14_or_23_quadrant, second_third_quadrant, atan)

    return result

register_op('atan2', atan2_symbolic, '', 11)  


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
    
def parse_config():    
    data_path="/media/charles/ShareDisk/00myDataSet/KITTI/archive/training/velodyne"
    cfg_ = '../output/cfgs/kitti_models/centerpoint/centerpoint.yaml'
    ckpt_ = '../output/cfgs/kitti_models/centerpoint/checkpoint_epoch_40.pth'
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_,
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default="demo",
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=ckpt_, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
            
def main():
    args, cfg = parse_config()
    
    logger = common_utils.create_logger()
    logger.info('------ Convert OpenPCDet model for TensorRT ------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # 打印模型
    print(model)
    
    onnx_model = onnx.load("xxxx.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx_model)
    
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            data_dict['encoded_spconv_tensor']

            # # 使用 ONNX Runtime 运行模型
            # ort_session = onnxruntime.InferenceSession("xxxx.onnx")

            # #构建输入并得到输出
            # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data_input)}
            # ort_outs = ort_session.run(None, ort_inputs)
            # ort_out = ort_outs[0]
            
            # np.testing.assert_allclose(to_numpy(torch_out), ort_out, rtol=1e-03, atol=1e-05)
    
    logger.info('Demo done.')

if __name__ == '__main__':
    main()
