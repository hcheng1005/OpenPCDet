import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate


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
        # if self.with_distance:
        #     num_point_features += 1

        self.num_point_features = num_point_features
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
    
    # def forward(self, batch_dict, **kwargs):
    #     voxel_features, voxel_num_points, coords  = batch_dict[0], batch_dict[1], batch_dict[2]
    #     points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
    #     f_cluster = voxel_features[:, :, :3] - points_mean
    #     f_center = torch.zeros_like(voxel_features[:, :, :3])

    #     # print("voxel_coords", coords[:,0])
    #     f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
    #     f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
    #     f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        
    #     if self.use_absolute_xyz:
    #         features = [voxel_features, f_cluster, f_center]
    #     else:
    #         features = [voxel_features[..., 3:], f_cluster, f_center]
    #     if self.with_distance:
    #         points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
    #         features.append(points_dist)
     
    #     features = torch.cat(features, dim=-1)
    #     voxel_count = features.shape[1]
    #     mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
    #     mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
    #     features *= mask
    #     for pfn in self.pfn_layers:
    #         features = pfn(features)
            
    #     features = features.squeeze()
    #     return features
    
    def forward(self, features, **kwargs):
  
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features[:,0,:]
        return features

def build_pfe(ckpt,cfg):
    pfe =PillarVFE(            
                model_cfg=cfg.MODEL.VFE,
                num_point_features=cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['NUM_POINT_FEATURES'],
                point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,  
                voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)  

    pfe.to('cuda').eval()

    checkpoint = torch.load(ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "vfe" in key:
            dicts[key[4:]] = checkpoint["model_state"][key]
    pfe.load_state_dict(dicts)
            
    # with torch.no_grad():
    #   MAX_VOXELS = 10000
    #   dummy_voxels = torch.zeros(
    #       (MAX_VOXELS, 32, 4),
    #       dtype=torch.float32,
    #       device='cuda:0')
    #   dummy_voxel_idxs = torch.zeros(
    #       (MAX_VOXELS, 4),
    #       dtype=torch.int32,
    #       device='cuda:0')
    #   dummy_voxel_num = torch.zeros(
    #       (1),
    #       dtype=torch.int32,
    #       device='cuda:0')

    #     # pytorch don't support dict when export model to onnx.
    #     # so here is something to change in networek input and output, the dict input --> list input
    #     # here is three part onnx export from OpenPCDet codebase:
    #   dummy_input = [dummy_voxels, dummy_voxel_num, dummy_voxel_idxs]
    # return pfe , dummy_input

    max_num_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_NUMBER_OF_VOXELS['test']
    max_points_per_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_POINTS_PER_VOXEL
    dims_feature = pfe.num_point_features
    dummy_input = torch.ones(max_num_pillars,max_points_per_pillars,dims_feature).cuda()
    return pfe , dummy_input 

if __name__ == "__main__":
    from pcdet.config import cfg, cfg_from_yaml_file
    
    cfg_file = "./cfgs/dual_radar_models/pointpillar_arbe.yaml"
    filename_mh = "./ckpt/dual_radar/pointpillars_arbe_100.pth"
    
    # cfg_file = "./cfgs/dual_radar_models/pointpillar_lidar.yaml"
    # filename_mh = "./ckpt/dual_radar/pointpillars_lidar_80.pth"
    
    cfg_from_yaml_file(cfg_file, cfg)
    model_cfg=cfg.MODEL
    pfe, dummy_input = build_pfe(filename_mh, cfg)
    pfe.eval().cuda()
    export_onnx_file = "./onnx_utils_dual_radar/arbe_pp_pfe.onnx"
    # torch.onnx.export(pfe,
    #                 dummy_input,
    #                 export_onnx_file,
    #                 opset_version=12,
    #                 verbose=True,
    #                 do_constant_folding=True,
    #                 input_names = ['voxel_features', 'voxel_num_points', 'coords'],   # the model's input names
    #                 output_names = ['features']) # the model's output names)  
    
    torch.onnx.export(pfe,
                    dummy_input,
                    export_onnx_file,
                    opset_version=12,
                    verbose=True,
                    do_constant_folding=True) # 输出名