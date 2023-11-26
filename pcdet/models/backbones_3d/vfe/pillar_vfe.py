import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        
        # x的维度由（M, 32, 10）升维成了（M, 32, 64）,max pool之后32才去掉
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        #permute变换维度，(M, 64, 32) --> (M, 32, 64)
          # 这里之所以变换维度，是因为BatchNorm1d在通道维度上进行,对于图像来说默认模式为[N,C,H*W],通道在第二个维度上
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        # 完成pointnet的最大池化操作，找出每个pillar中最能代表该pillar的点
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        # num_point_features:10
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        #[64]
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        # num_filters:  [10, 64]
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        #len(num_filters) - 1 == 1
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i] # 10
            out_filters = num_filters[i + 1] # 64
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        #收集PFN层，在forward中执行
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
        '''
        指出一个pillar中哪些是真实数据,哪些是填充的0数据
        '''
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):

        '''
	batch_dict:
            points:(N,5) --> (batch_index,x,y,z,r) batch_index代表了该点云数据在当前batch中的index
            frame_id:(batch_size,) -->帧ID-->我们存放的是npy的绝对地址，batch_size个地址
            gt_boxes:(batch_size,N,8)--> (x,y,z,dx,dy,dz,ry,class)，
            use_lead_xyz:(batch_size,) --> (1,1,1,1)，batch_size个1
            voxels:(M,32,4) --> (x,y,z,r)
            voxel_coords:(M,4) --> (batch_index,z,y,x) batch_index代表了该点云数据在当前batch中的index
            voxel_num_points:(M,):每个voxel内的点云
            batch_size:4：batch_size大小
        '''
  
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        #求每个pillar中所有点云的平均值,设置keepdim=True的，则保留原来的维度信息
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        #每个点云数据减去该点对应pillar的平均值，得到差值 xc,yc,zc
        f_cluster = voxel_features[:, :, :3] - points_mean

        # 创建每个点云到该pillar的坐标中心点偏移量空数据 xp,yp,zp
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        '''
          coords是每个网格点的坐标，即[432, 496, 1]，需要乘以每个pillar的长宽得到点云数据中实际的长宽（单位米）
          同时为了获得每个pillar的中心点坐标，还需要加上每个pillar长宽的一半得到中心点坐标
          每个点的x、y、z减去对应pillar的坐标中心点，得到每个点到该点中心点的偏移量
        '''
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        #配置中使用了绝对坐标，直接组合即可。
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center] #10个特征，直接组合
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        #距离信息，False
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        #mask中指明了每个pillar中哪些是需要被保留的数据
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        #由0填充的数据，在计算出现xc,yc,zc和xp,yp,zp时会有值
        #features中去掉0值信息。
        features *= mask
        #执行上面收集的PFN层，每个pillar抽象出64维特征
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
