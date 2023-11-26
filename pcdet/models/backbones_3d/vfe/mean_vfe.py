import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features # 5

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        # [num_voxels,10,5],[num_voxels]
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        # keepdim参数指是否对求和的结果squeeze,如果True其他维度保持不变，求和的dim维变为1，False删除维度
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False) # 第二维相加[num_voxels,5]
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer # [num_voxels,5]
        batch_dict['voxel_features'] = points_mean.contiguous() # 深拷贝

        return batch_dict
