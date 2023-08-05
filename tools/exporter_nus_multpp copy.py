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

from pathlib import Path
from onnxsim import simplify
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file

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
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_pp_multihead_demo.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='ckpt/nuScenes/pp_multihead_nds5823_updated.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    # export_paramters(cfg)
    logger = common_utils.create_logger()
    logger.info('------ Convert OpenPCDet model for TensorRT ------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    # model.to('cpu')
    model.cuda()
    model.eval()
    
    # 
    print(model)
    
    np.set_printoptions(threshold=np.inf)
    with torch.no_grad():

      MAX_VOXELS = 10000

      dummy_voxels = torch.zeros(
          (MAX_VOXELS, 32, 5),
          dtype=torch.float32,
          device='cuda:0')

      dummy_voxel_idxs = torch.zeros(
          (MAX_VOXELS, 5),
          dtype=torch.int32,
          device='cuda:0')

      dummy_voxel_num = torch.zeros(
          (1),
          dtype=torch.int32,
          device='cuda:0')

      dummy_input = dict()
      dummy_input['voxels'] = dummy_voxels
      dummy_input['voxel_num_points'] = dummy_voxel_num
      dummy_input['voxel_coords'] = dummy_voxel_idxs
      dummy_input['batch_size'] = 1
      
      torch.onnx.export(model,       # model being run
          dummy_input,               # model input (or a tuple for multiple inputs)
          "./pp.onnx",  # where to save the model (can be a file or file-like object)
          export_params=True,        # store the trained parameter weights inside the model file
          opset_version=11,          # the ONNX version to export the model to
          do_constant_folding=False,  # whether to execute constant folding for optimization
          keep_initializers_as_inputs=True,
          input_names = ['voxels', 'voxel_num', 'voxel_idxs'],   # the model's input names
          output_names = ['cls_preds', 'box_preds', 'dir_cls_preds'], # the model's output names
          )

    onnx_raw = onnx.load("./pp.onnx")  # load onnx model
    onnx_simp, check = simplify(onnx_raw)
    onnx.save(onnx_simp, "pp_simple.onnx")

    onnx_raw = onnx.load("./pp_simple.onnx")  # load onnx model
    onnx_trim_post = simplify_postprocess(onnx_raw)
    onnx.save(onnx_trim_post, "pp_simple2.onnx")
      
    # onnx_simp, check = simplify(onnx_trim_post)
    # onnx.save(onnx_simp, "pp_simple3.onnx")
    # assert check, "Simplified ONNX model could not be validated"

    onnx_final = simplify_preprocess(onnx_trim_post)
    onnx.save(onnx_final, "multpp_final.onnx")
    print('finished exporting onnx')

    logger.info('[PASS] ONNX EXPORTED.')

if __name__ == '__main__':
    main()
