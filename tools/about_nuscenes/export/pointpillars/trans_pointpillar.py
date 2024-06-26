'''
Date: 2023-12-28 20:30:24
LastEditors: CharlesHAO hao.cheng@wuzheng.com
LastEditTime: 2024-02-29 19:50:13
FilePath: /OpenPCDet/tools/about_nuscenes/export/pointpillars/trans_pointpillar.py
'''
import os
import argparse

import glob
import onnx
import argparse
import numpy as np

from pathlib import Path
from onnxsim import simplify

from torchinfo import summary

import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate
from onnx_backbone_2d import BaseBEVBackbone
from onnx_dense_head2 import  AnchorHeadMulti
from pcdet.config import cfg, cfg_from_yaml_file

from exporter_paramters import export_paramters as export_paramters
from simplifier_nus_multpp_onnx import simplify_preprocess, simplify_postprocess

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=True):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.part = 50000

    def forward(self, inputs):
        # nn.Linear performs randomly when batch size is too large
        x = self.linear(inputs)

        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        return x_max

class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
            
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
    
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):

        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)#len(actual_num.shape) 求actual_num.shape的维度
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator
    
    def forward(self, voxel_features, voxel_num_points, coords, **kwargs):
        # voxel_features, voxel_num_points, coords  = batch_dict[0], batch_dict[1], batch_dict[2]
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean
        f_center = torch.zeros_like(voxel_features[:, :, :3])

        # print("voxel_coords", coords[:,0])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
     
        features = torch.cat(features, dim=-1)
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze()
        return features

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES #64
        self.nx, self.ny, self.nz = grid_size # [432,496,1]
        assert self.nz == 1

    def forward(self, pillar_features, coords, **kwargs):
        '''
        batch_dict['pillar_features']-->为VFE得到的数据(M, 64)
        voxel_coords:(M,4) --> (batch_index,z,y,x) batch_index代表了该点云数据在当前batch中的index
        '''
        # pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        # 根据batch_index，获取batch_size大小
        batch_size = 1
        
        spatial_feature = torch.zeros(
            self.num_bev_features,
            self.nz * self.nx * self.ny,
            dtype=pillar_features.dtype,
            device=pillar_features.device)

        batch_mask = coords[:, 0] == 0 #返回mask，[True, False...]
        this_coords = coords[batch_mask, :] #获取当前的batch_idx的数
        indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
        indices = indices.type(torch.long)  # 转换数据类型
        pillars = pillar_features[batch_mask, :]
        pillars = pillars.t()
        spatial_feature[:, indices] = pillars
        
        # reshape回原空间(伪图像)    （4, 64, 214272）--> (4, 64, 496, 432)
        spatial_feature = spatial_feature.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        # batch_dict['spatial_features'] = batch_spatial_features
        #返回数据
        return spatial_feature

'''
name: pointpillars
description: 重新定义pointpillars模型
return {*}
'''
class pointpillars(nn.Module):
    def __init__(self, cfg, grid_size):
        super().__init__()
        '''
        以下模型全都重新定义 PillarVFE PointPillarScatter BaseBEVBackbone AnchorHeadMulti
        '''
        self.vfe = PillarVFE(model_cfg=cfg.MODEL.VFE,
                            num_point_features=cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['NUM_POINT_FEATURES'],
                            point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,  
                            voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)  
        
        self.map_to_bev = PointPillarScatter(cfg.MODEL.MAP_TO_BEV, grid_size)
        
        self.backbone_2d = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D, 64)
        
        self.dense_head = AnchorHeadMulti(model_cfg=cfg.MODEL.DENSE_HEAD,
                                            input_channels=384,
                                            num_class=len(cfg.CLASS_NAMES),
                                            class_names=cfg.CLASS_NAMES,
                                            grid_size= grid_size,
                                            point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                                            predict_boxes_when_training=False)

    def forward(self, voxel_features, voxel_num_points, coords):
        features = self.vfe.forward(voxel_features, voxel_num_points, coords)
        spatial_features = self.map_to_bev.forward(features, coords)
        spatial_features_2d = self.backbone_2d.forward(spatial_features)
        box_preds, cls_preds = self.dense_head.forward(spatial_features_2d)
        return box_preds, cls_preds


'''
name: build_pointpillars
description: 由于新版openpcdet无法直接使用onnx导出模型，因此这里重新定义point pillars模型
param {*} ckpt
param {*} cfg
return {*}
'''
def build_pointpillars(ckpt,cfg):
    grid_size = np.array([512, 512, 1])
    model = pointpillars(cfg, grid_size)  

    model.to('cuda').eval()

    checkpoint = torch.load(ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "vfe" in key:
            dicts[key] = checkpoint["model_state"][key]
        if "backbone_2d" in key:
            dicts[key] = checkpoint["model_state"][key]
        if "dense_head" in key:
            dicts[key] = checkpoint["model_state"][key]
    model.load_state_dict(dicts)
            
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

        # pytorch don't support dict when export model to onnx.
        # so here is something to change in networek input and output, the dict input --> list input
        # here is three part onnx export from OpenPCDet codebase:
      dummy_input = (dummy_voxels, dummy_voxel_num, dummy_voxel_idxs)
    return model , dummy_input

if __name__ == "__main__":
    from pcdet.config import cfg, cfg_from_yaml_file
    
    cfg_file = './cfgs/nuscenes_models/cbgs_pp_multihead.yaml'
    filename_mh = "./ckpt/nuScenes/pp_multihead_nds5823_updated.pth"
    
    cfg_from_yaml_file(cfg_file, cfg)
    export_paramters(cfg)
    model_cfg=cfg.MODEL
    model, dummy_input = build_pointpillars(filename_mh, cfg)
    # summary(model, input_size=[(10000, 32, 5), (1,), (10000, 5)], depth = 5)
    # summary(model, input_data=dummy_input)
    
    model.eval().cuda()
    export_onnx_file = "./about_nuscenes/export/pointpillars/output/nus_pp_mutiHead_raw.onnx"
    
    # 导出pp模型
    torch.onnx.export(model,
                    dummy_input,
                    export_onnx_file,
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    keep_initializers_as_inputs=True,
                    input_names = ['voxels', 'voxel_num', 'voxel_idxs'],   # the model's input names
                    output_names = ['cls_preds', 'box_preds'])# the model's output names
    
    onnx_raw = onnx.load("./about_nuscenes/export/pointpillars/output/nus_pp_mutiHead_raw.onnx")  # load onnx model
    onnx_trim_post = simplify_postprocess(onnx_raw) # 简化后处理
    
    onnx_simp, check = simplify(onnx_trim_post) # 模型简化（库函数）
    assert check, "Simplified ONNX model could not be validated"

    onnx_final = simplify_preprocess(onnx_simp) # 前处理简化
    onnx.save(onnx_final, "./about_nuscenes/export/pointpillars/output/nus_pp_mutiHead.onnx")
    print('finished exporting onnx')