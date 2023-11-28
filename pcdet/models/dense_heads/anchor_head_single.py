import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    '''
    Args:
        model_cfg: AnchorHeadSingle的配置
        input_channels: 384 输入通道数
        num_class: 3
        class_names: ['Car','Pedestrian','Cyclist']
        grid_size: (432, 496, 1)
        point_cloud_range: (0, -39.68, -3, 69.12, 39.68, 1)
        predict_boxes_when_training: False
    '''
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        #在父类中调用generate_anchors中生成anchors和num_anchors_per_location
        # 每个点会生成不同类别的2个先验框(anchor)，也就是说num_anchors_per_location：[2, 2, 2,]-》3类，每类2个anchor
        #所以每个点生成6个先验框(anchor)
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        #类别， 1x1 卷积：conv_cls:  Conv2d(384, 18, kernel_size=(1, 1), stride=(1, 1))
        #每个点6个anchor，每个anchor预测3个类别，所以输出的类别为6*3
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        #box，1x1 卷积：conv_box:  Conv2d(384, 42, kernel_size=(1, 1), stride=(1, 1))
        #每个点6个anchor，每个anchor预测7个值（x, y, z, w, l, h, θ），所以输出的值为6*7
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size, #self.box_coder.code_size默认为7
            kernel_size=1
        )
        # 是否使用方向分类，1x1 卷积：conv_dir:  Conv2d(384, 12, kernel_size=(1, 1), stride=(1, 1))
        #每个点6个anchor，每个anchor预测2个方向(正负)，所以输出的值为6*2
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()
    
    #初始化参数
    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        # spatial_features_2d 维度 ：（batch_size, 384, 248, 216）
        spatial_features_2d = data_dict['spatial_features_2d']

        #cls_preds的维度为：torch.Size([batch_size, 18, 248, 216])
        #每个点6个anchor，每个anchor预测3个类别，所以输出的类别为6*3
        cls_preds = self.conv_cls(spatial_features_2d)
        #box_preds的维度为：torch.Size([batch_size, 42, 248, 216])
        #每个点6个anchor，每个anchor预测7个值（x, y, z, w, l, h, θ），所以输出的值为6*7
        box_preds = self.conv_box(spatial_features_2d)

        #调整顺序
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        #方向预测，也就是正负预测
        if self.conv_dir_cls is not None:
            #dir_cls_preds的维度为：torch.Size([batch_size, 12, 248, 216])
            #每个点6个anchor，每个anchor预测2个方向(正负)，所以输出的值为6*2
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            #调整顺序
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        #如果是在训练模式的时候，需要对每个先验框分配GT来计算loss
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            #分配的
            self.forward_ret_dict.update(targets_dict)
        #非训练模式，则直接生成进行box的预测
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
