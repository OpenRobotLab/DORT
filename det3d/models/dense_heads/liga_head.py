import numpy as np
import torch
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from torch import nn as nn

import torch.nn.functional as F
from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from mmdet.models import HEADS, build_loss
from det3d.models.necks.liga_neck import convbn, convbn_3d


from mmdet3d.models.dense_heads import Anchor3DHead

@HEADS.register_module()
class LigaDepthHead(nn.Module):

    def __init__(self,
                  num_hg=1,
                  cv_dim=32,
                  downsample_disp=4,
                  disp_regression=True,
                  use_GN=True):
        super().__init__()
        self.num_hg = num_hg
        self.disp_regression=disp_regression

        self.pred_stereo = nn.ModuleList()
        for _ in range(self.num_hg):
            self.pred_stereo.append(
                nn.Sequential(
                    convbn_3d(cv_dim, cv_dim, 3, 1, 1, gn=use_GN),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(cv_dim, 1, 3, 1, 1, bias=False),
                    nn.Upsample(scale_factor=downsample_disp,
                                 mode='trilinear', align_corners=True)))

    @torch.no_grad()
    def get_local_depth(self, d_prob, depth):
        d = depth.cuda()[None, :, None, None]
        d_mul_p = d * d_prob
        local_window = 5
        p_local_sum = 0
        for off in range(0, local_window):
            cur_p = d_prob[:, off:off + d_prob.shape[1] - local_window + 1]
            p_local_sum += cur_p
        max_indices = p_local_sum.max(1, keepdim=True).indices
        pd_local_sum_for_max = 0
        for off in range(0, local_window):
            cur_pd = torch.gather(d_mul_p, 1, max_indices + off).squeeze(1)  # d_prob[:, off:off + d_prob.shape[1] - local_window + 1]
            pd_local_sum_for_max += cur_pd
        mean_d = pd_local_sum_for_max / torch.gather(p_local_sum, 1, max_indices).squeeze(1)
        return mean_d

    def forward(self, all_costs, depth, image_shape):
        assert len(all_costs) == len(self.pred_stereo)
        cost_list = []
        cost_softmax_list = []
        pred_list = []
        depth_preds_local_list = []
        for cost, depth_conv_module in zip(all_costs, self.pred_stereo):
            cost1 = depth_conv_module(cost)
            cost1 = torch.squeeze(cost1, 1)

            cost_softmax = F.softmax(cost1, dim=1)
            if self.disp_regression:
                pred1 = self.depth_to_disp(cost_softmax, depth=depth)

            if not self.training:
                depth_preds_local = self.get_local_depth(cost_softmax, depth)
                depth_preds_local_list.append(depth_preds_local)
            cost_list.append(cost1)
            cost_softmax_list.append(cost_softmax)
            pred_list.append(pred1)

        return pred_list, cost_list, cost_softmax_list, depth_preds_local_list


    def depth_to_disp(self, x, depth):
        assert len(x.shape) == 4
        assert len(depth.shape) == 1
        out = torch.sum(x * depth[None, :, None, None], 1)
        return out


@HEADS.register_module()
class LigaDetHead(Anchor3DHead):
    """
    Modification: replace the network architecure to two conv layer as liga.
    Anchor head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 feat_channels=256,
                 use_direction_classifier=True,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[1.6, 3.9, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size=False,
                 assign_per_class=False,
                 diff_rad_by_sin=True,
                 dir_offset=0,
                 dir_limit_offset=1,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2),
                 num_convs=2,
                 use_GN=True,
                 init_cfg=None):

        self.use_GN = use_GN
        self.num_convs = num_convs
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            feat_channels=feat_channels,
            use_direction_classifier=use_direction_classifier,
            anchor_generator=anchor_generator,
            assigner_per_size=assigner_per_size,
            assign_per_class=assign_per_class,
            diff_rad_by_sin=diff_rad_by_sin,
            dir_offset=dir_offset,
            dir_limit_offset=dir_limit_offset,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            init_cfg=init_cfg)

    def _init_layers(self):

        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.rpn3d_cls_convs = []
        self.rpn3d_bbox_convs = []
        if self.num_convs > 0:
            for _ in range(self.num_convs):
                self.rpn3d_cls_convs.append(nn.Sequential(
                    convbn(self.feat_channels, self.feat_channels,
                             3, 1, 1, 1, gn=self.use_GN),
                    nn.ReLU(inplace=True)))
                
                self.rpn3d_bbox_convs.append(nn.Sequential(
                    convbn(self.feat_channels, self.feat_channels,
                             3, 1, 1, 1, gn=self.use_GN),
                    nn.ReLU(inplace=True)))
            self.rpn3d_cls_convs = nn.Sequential(*self.rpn3d_cls_convs)
            self.rpn3d_bbox_convs = nn.Sequential(*self.rpn3d_bbox_convs)
        else:
            self.rpn3d_cls_convs = nn.Identity()
            self.rpn3d_bbox_convs = nn.Identity()
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                          self.num_anchors * 2, 1)

    
    def forward_single(self, x):
        cls_feats = self.rpn3d_cls_convs(x)
        bbox_feats = self.rpn3d_bbox_convs(x)
        cls_score = self.conv_cls(cls_feats)
        bbox_pred = self.conv_reg(bbox_feats)
        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(bbox_feats)
        
        return cls_score, bbox_pred, dir_cls_preds
