import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES #64
        self.nx, self.ny, self.nz = grid_size # [432,496,1]
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        '''
        batch_dict['pillar_features']-->为VFE得到的数据(M, 64)
        voxel_coords:(M,4) --> (batch_index,z,y,x) batch_index代表了该点云数据在当前batch中的index
        '''
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        # 根据batch_index，获取batch_size大小
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            # 创建一个空间坐标所有用来接受pillar中的数据
            # spatial_feature 维度 (64,214272)
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx #返回mask，[True, False...]
            this_coords = coords[batch_mask, :] #获取当前的batch_idx的数
            #计算pillar的索引，该点之前所有行的点总和加上该点所在的列即可
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)  # 转换数据类型
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64, 214272)
            batch_spatial_features.append(spatial_feature)

        # 在第0个维度将所有的数据堆叠在一起
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
         # reshape回原空间(伪图像)    （4, 64, 214272）--> (4, 64, 496, 432)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        #返回数据
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
    
    
    
class RadarPointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        # 获取毫米波雷达信息
        pillar_features, coords = batch_dict['pillar_features_radar'], batch_dict['voxel_coords_radar']
        batch_spatial_features = []
        # print(coords.shape)  [28030, 4]
        # print(coords[:, 0])
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        
        # 返回毫米波雷达特征信息
        batch_dict['spatial_features_radar'] = batch_spatial_features
        return batch_dict