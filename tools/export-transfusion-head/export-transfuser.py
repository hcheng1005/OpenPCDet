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

import sys; sys.path.insert(0, "./")

# from torchpack.utils.config import configs
# from mmcv.cnn import fuse_conv_bn
# from mmcv import Config
# from mmcv.runner.fp16_utils import auto_fp16
# from mmdet3d.models import build_model
# from mmdet3d.utils import recursive_eval
# from mmcv.runner import wrap_fp16_model

import copy

from pcdet.utils import common_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

from torch.nn.init import kaiming_normal_

from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file

from pcdet.models.model_utils.basic_block_2d import BasicBlock2D
from pcdet.models.model_utils.transfusion_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from pcdet.models.dense_heads import TransFusionHead

import collections


from pathlib import Path
import numpy as np
import glob

'''
names: 
description: Briefly describe the function of your function
return {*}
'''
class SeparateHead_Transfusion(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size//2, bias=use_bias),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size//2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class SubclassBEVFusionFuserDecoder(nn.Module):

    def __init__(
        self,
        model_cfg, 
        input_channels, 
        num_class, 
        class_names, 
        grid_size, 
        point_cloud_range, 
        voxel_size, 
        predict_boxes_when_training=True,
    ):
        super(SubclassBEVFusionFuserDecoder, self).__init__()
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = num_class

        self.model_cfg = model_cfg
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')

        hidden_channel=self.model_cfg.HIDDEN_CHANNEL
        self.num_proposals = self.model_cfg.NUM_PROPOSALS
        self.bn_momentum = self.model_cfg.BN_MOMENTUM
        self.nms_kernel_size = self.model_cfg.NMS_KERNEL_SIZE

        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.DROPOUT
        activation = self.model_cfg.ACTIVATION
        ffn_channel = self.model_cfg.FFN_CHANNEL
        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)

        # loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        # self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        # if not self.use_sigmoid_cls:
        #     self.num_classes += 1
        # self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        # self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        # self.loss_bbox = loss_utils.L1Loss()
        # self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        # self.loss_heatmap = loss_utils.GaussianFocalLoss()
        # self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']

        self.code_size = 10

        # a shared convolution
        self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        layers = []
        layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=self.num_classes,kernel_size=3,padding=1))
        
        # heatmap特征卷积网络
        self.heatmap_head = nn.Sequential(*layers)
        # class类型卷积网络
        self.class_encoding = nn.Conv1d(self.num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel))
        
        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)

        # self.init_weights()
        # self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.forward_ret_dict = {}

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base
    
    @staticmethod
    # @auto_fp16(apply_to=("inputs", "classes_eye"))
    def head_forward(self, inputs):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = 1# inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )  # [BS, C, H*W]
        bev_pos = self.bev_pos.to(lidar_feat.dtype).repeat(batch_size, 1, 1).to(lidar_feat.device)

        #################################
        # image guided query initialization
        #################################
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            # local_max[
            #     :,
            #     8,
            # ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            # local_max[
            #     :,
            #     9,
            # ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            local_max[:, 8] = heatmap[:, 8]
            local_max[:, 9] = heatmap[:, 9]
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            # local_max[
            #     :,
            #     1,
            # ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            # local_max[
            #     :,
            #     2,
            # ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            local_max[:, 1] = heatmap[:, 1]
            local_max[:, 2] = heatmap[:, 2]
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top #num_proposals among all classes
        # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
        #     ..., : self.num_proposals
        # ]
        top_proposals = heatmap.view(batch_size, -1).topk(k=self.num_proposals, dim=-1, largest=True)[1]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1
            ),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        self.one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
            0, 2, 1)
        # self.one_hot = classes_eye.index_select(0, top_proposals_class.view(-1))[None].permute(
        #     0, 2, 1
        # )
        query_cat_encoding = self.class_encoding(self.one_hot)
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]
            .permute(0, 2, 1)
            .expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        # ret_dicts = []
        # for i in range(self.num_decoder_layers):
        # prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

        # Transformer Decoder Layer
        # :param query: B C Pq    :param query_pos: B Pq 3/6
        query_feat = self.decoder(
            query_feat, lidar_feat_flatten, query_pos, bev_pos
        )

        # Prediction
        res_layer = self.prediction_heads(query_feat)
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
        # first_res_layer = res_layer
        # ret_dicts.append(res_layer)

        # for next level positional embedding
        # query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        res_layer["dense_heatmap"] = dense_heatmap

        # # if self.auxiliary is False:
        # #     # only return the results of last decoder layer
        # #     return ret_dicts[-1]

        # # return all the layer's results for auxiliary superivison
        # new_res = {}
        # for key in ret_dicts[0].keys():
        #     if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
        #         new_res[key] = torch.cat(
        #             [ret_dict[key] for ret_dict in ret_dicts], dim=-1
        #         )
        #     else:
        #         new_res[key] = ret_dicts[0][key]
        return res_layer

    def get_bboxes(self, preds_dict, one_hot):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        # batch_score = preds_dict["heatmap"][..., -self.num_proposals :].sigmoid()
        batch_score = preds_dict["heatmap"].sigmoid()
        # if self.loss_iou.loss_weight != 0:
        #    batch_score = torch.sqrt(batch_score * preds_dict['iou'][..., -self.num_proposals:].sigmoid())
        one_hot = F.one_hot(
            self.query_labels, num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * preds_dict["query_heatmap_score"] * one_hot
        # batch_center = preds_dict["center"][..., -self.num_proposals :]
        # batch_height = preds_dict["height"][..., -self.num_proposals :]
        # batch_dim = preds_dict["dim"][..., -self.num_proposals :]
        # batch_rot = preds_dict["rot"][..., -self.num_proposals :]
        # batch_vel = None
        # if "vel" in preds_dict:
        #     batch_vel = preds_dict["vel"][..., -self.num_proposals :]
        batch_center = preds_dict["center"]
        batch_height = preds_dict["height"]
        batch_dim = preds_dict["dim"]
        batch_rot = preds_dict["rot"]
        batch_vel = None
        if "vel" in preds_dict:
            batch_vel = preds_dict["vel"]

        return [batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel]

    # @auto_fp16(apply_to=("features",))
    def forward(self, features):
        # if self.parent.fuser is not None:
        #     x = self.parent.fuser(features)
        # else:
        #     assert len(features) == 1, features
        #     x = features[0]

        # batch_size = x.shape[0]
        # x = self.parent.decoder["backbone"](x)
        # x = self.parent.decoder["neck"](x)

        # outputs = [{} for _ in range(batch_size)]
        # for type, head in self.parent.heads.items():
        #     if type == "object":
        #         pred_dict = self.head_forward(head, x[0], self.classes_eye)
        #         return self.get_bboxes(pred_dict, head.one_hot)
        #     else:
        #         raise ValueError(f"unsupported head: {type}")
            
        # return outputs  
    
        pred_dict = self.head_forward(self, features)
        # return self.get_bboxes(pred_dict, head.one_hot)
                
        


def parse_config():
    # transFusion
    data_path="/root/code/data/nuscenes/sweeps/LIDAR_TOP/"
    cfg_ = '../cfgs/nuscenes_models/transfusion_lidar.yaml'
    ckpt_ = '../ckpt/cbgs_transfusion_lidar.pth'
        
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_,
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=ckpt_,
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

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
        data_file_list = glob.glob(
            str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        points = np.fromfile(
            self.sample_file_list[index], dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

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



if __name__ == "__main__":
    
    args, cfg = parse_config()
    
    # half_export = True
    # if not os.path.exists(args.input):
    #     camera_features = torch.randn(1, 80, 180, 180).cuda()
    #     lidar_features  = torch.randn(1, 256, 180, 180).cuda()
    # else:
    #     camera_features, lidar_features = torch.load(args.input)
    #     camera_features = camera_features.cuda()
    #     lidar_features  = lidar_features.cuda()

    # lidar_features  = torch.randn(1, 256, 180, 180).cuda()
    
    # os.makedirs(os.path.dirname(args.save_onnx), exist_ok=True)
    # configs.load(args.config, recursive=True)
    # cfg = Config(recursive_eval(configs), filename=args.config)

    # cfg.model.train_cfg = None
    
    logger = common_utils.create_logger()
    logger.info(
        '-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    
    model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': demo_dataset.point_feature_encoder.num_point_features,
            'num_point_features': demo_dataset.point_feature_encoder.num_point_features,
            'grid_size': demo_dataset.grid_size,
            'point_cloud_range': demo_dataset.point_cloud_range,
            'voxel_size': demo_dataset.voxel_size,
            'depth_downsample_factor':demo_dataset.depth_downsample_factor
        }
            
    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=demo_dataset)
        
    model = SubclassBEVFusionFuserDecoder(
            model_cfg=cfg.MODEL.DENSE_HEAD,
            input_channels= 512,
            num_class=len(cfg.CLASS_NAMES),
            class_names=cfg.CLASS_NAMES,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=False,
            voxel_size=model_info_dict.get('voxel_size', False)
        )
    
    checkpoint = torch.load(args.ckpt, map_location='cuda')["model_state"]
    
    # print(checkpoint["model_state"].keys())
    new_ckpt = collections.OrderedDict()
    for key, val in checkpoint.items():
        # print(key)
        if key.startswith("dense_head."):
            print(key)
            newkey = key[key.find(".")+1:]
            new_ckpt[newkey] = val
        
    model.load_state_dict(new_ckpt)
    
    print(model)
    
    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES))

    # if args.ckpt:
    #   model = funcs.load_checkpoint(model, args.ckpt)

    # # model = fuse_conv_bn(model)
    # model = SubclassBEVFusionFuserDecoder(model).cuda()
    # model.float()
    
    # lidar_features  = lidar_features.float()
    lidar_features  = torch.randn(1, 128, 180, 180).cuda()
    

    # if half_export:
    #     wrap_fp16_model(model)
    #     camera_features = camera_features.float()
    #     lidar_features  = lidar_features.float()
    # else:
    #     model.float()
    #     camera_features = camera_features.float()
    #     lidar_features  = lidar_features.float()

    with torch.no_grad():
        torch.onnx.export(model, lidar_features, "./transfusion_head.onnx", opset_version=14, 
            input_names=["camera", "lidar"],
            output_names=["score", "rot", "dim", "reg", "height", "vel"],
            # dynamic_axes={"hm": {0: "batch"},"rot": {0: "batch"},"dim": {0: "batch"},"reg": {0: "batch"},"height": {0: "batch"},"vel": {0: "batch"}, "camera": {0: "batch"}, "lidar": {0: "batch"}}
        )
    print(f"Save onnx to '{'transfusion_head.onnx'}'")
    print("Export to ONNX is complete. 🤗")
