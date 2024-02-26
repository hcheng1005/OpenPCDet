import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate
from onnx_backbone_2d import BaseBEVBackbone
# from onnx_dense_head import SingleHead

from pcdet.models.dense_heads import AnchorHeadTemplate
# from pcdet.models.dense_heads import AnchorHeadSingle

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        """
        Args:
            model_cfg: AnchorHeadSingle的配置
            input_channels:384 输入通道数
            num_class: 3
            class_names: ['Car','Pedestrian','Cyclist']
            grid_size: (432,493,1)
            point_cloud_range:(0, -39.68, -3, 69.12, 39.68, 1)
            predict_boxes_when_training:False
        """
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location) # 2*3=6
        # Conv2d(384,18,kernel_size=(1,1),stride=(1,1))
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class, # 6*3=18
            kernel_size=1
        )
        # Conv2d(384,42,kernel_size=(1,1),stride=(1,1))
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size, # 6*7=42
            kernel_size=1
        )
        # 如果存在方向损失，则添加方向卷积层Conv2d(384,12,kernel_size=(1,1),stride=(1,1))
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS, # 6*2=12
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        # 参数初始化
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, spatial_features_2d):
        # spatial_features_2d = data_dict['spatial_features_2d'] # （4，384，248，216）
        cls_preds = self.conv_cls(spatial_features_2d) # 每个anchor的类别预测-->(4,18,248,216)
        box_preds = self.conv_box(spatial_features_2d) # 每个anchor的box预测-->(4,42,248,216)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] -->(4,248,216,42)
        dir_cls_preds = self.conv_dir_cls(spatial_features_2d) # 每个anchor的方向预测-->(4,12,248,216)
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] -->(4,248,216,12)
        return box_preds, cls_preds, dir_cls_preds
    
    

class backbone_singlehead(nn.Module):
    def __init__(self, cfg, grid_size):
        super().__init__()
        self.backbone_2d = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D, 64)
        self.dense_head =  AnchorHeadSingle(model_cfg=cfg.MODEL.DENSE_HEAD,
                                            input_channels=384,
                                            num_class=len(cfg.CLASS_NAMES),
                                            class_names=cfg.CLASS_NAMES,
                                            grid_size=grid_size,
                                            point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                                            predict_boxes_when_training=False)

    def forward(self, spatial_features):
        spatial_features_2d = self.backbone_2d.forward(spatial_features)
        box_preds, cls_preds, dir_cls_preds = self.dense_head.forward(spatial_features_2d)
        return box_preds, cls_preds, dir_cls_preds


def build_backbone_singlehead(ckpt,cfg):
    grid_size = np.array([496, 640, 1])
    model = backbone_singlehead(cfg, grid_size)
    model.to('cuda').eval()
    # print(model)
   
    checkpoint = torch.load(ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "backbone_2d" in key:
            dicts[key] = checkpoint["model_state"][key]
        if "dense_head" in key:
            dicts[key] = checkpoint["model_state"][key]
    model.load_state_dict(dicts)
    
    dummy_input = torch.ones(1, 64, grid_size[0], grid_size[1]).cuda()
    return model , dummy_input

if __name__ == "__main__":
    from pcdet.config import cfg, cfg_from_yaml_file

    cfg_file = "./cfgs/dual_radar_models/pointpillar_arbe.yaml"
    filename_mh = "./ckpt/dual_radar/pointpillars_arbe_100.pth"    
    
    # cfg_file = "./cfgs/dual_radar_models/pointpillar_liadr.yaml"
    # filename_mh = "./ckpt/dual_radar/pointpillars_lidar_80.pth"
    
    cfg_from_yaml_file(cfg_file, cfg)
    model, dummy_input = build_backbone_singlehead(filename_mh, cfg)
    model.eval().cuda()
    export_onnx_file = "./onnx_utils_dual_radar/arbe_pp_backbone2.onnx"
    torch.onnx.export(model,
                    dummy_input,
                    export_onnx_file,
                    opset_version=10, # v10 is better than v12
                    verbose=True,
                    do_constant_folding=True,
                    input_names = ['features'],   # the model's input names
                    output_names = ['box_preds', 'cls_preds', 'dir_cls_preds']) # the model's output names)   # the model's input names) # 输出名
    
    # torch.onnx.export(model,
    #                   dummy_input,
    #                   export_onnx_file,
    #                   opset_version=10,
    #                   verbose=True,
    #                   do_constant_folding=True) # 输出名
