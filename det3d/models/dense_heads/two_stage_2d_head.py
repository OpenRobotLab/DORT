import torch.nn.functional as F
import torch


from mmdet.models.builder import HEADS, build_loss

from mmdet3d.core.bbox.structures import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes
import torch.nn as nn
from .centernet_utils import CenterNetHeatMap, CenterNetDecoder, decode_dimension, gather_feature
from det3d.core.bbox.util import projected_2d_box, alpha_to_ry, points_img2cam, points_img2cam_batch


# from inplace_abn import InPlaceABN
# from .loss_utils import reg_l1_loss, FocalLoss, depth_uncertainty_loss, BinRotLoss
import math
from mmdet3d.core.bbox.structures import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps

from .loss_utils import compute_rot_loss
from .centernet3d_head import get_orientation_bin, get_alpha
# from mmdet3d.core.bbox.structures.cam_box3d import 
from det3d.models.necks.imvoxel_neck import BasicBlock2d
from mmcv import ops

from .two_stage_head import TwoStageHead


@HEADS.register_module()
class TwoStage2DHead(TwoStageHead):
    def __init__(self, num_classes=3,
                        box_code_size=6,
                        input_channels=64,
                        feat_channels=64,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=True,
                            loss_weight=1.0),
                        loss_bbox=dict(
                            type='SmoothL1Loss', beta=1.0/9.0, loss_weight=2.0),
                        loss_dir=dict(type='BinRotLoss'),
                        iou3d_thresh=0.2,
                        iou2d_thresh=0.5,
                        insert_noise_with_gt=True,
                        roi_config = dict(
                            type="RoIAlign", output_size=14,
                            spatial_scale=0.25, sampling_ratio=0),
                        noise_scale=[0.2, 0.2, 0.2, 0.1, 0., 0., 0.],
                        refine_proposal=True):

        super().__init__(num_classes,
                        box_code_size,
                        iou3d_thresh,
                        iou2d_thresh,
                        noise_scale,
                        refine_proposal=refine_proposal)

        self.input_channels = input_channels
        self.feat_channels = feat_channels

        self.insert_noise_with_gt=insert_noise_with_gt

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        if loss_dir["type"] == "BinRotLoss":
            self.loss_dir = loss_dir # TODO modify it to mmdet3d family
        else:
            self.loss_dir = build_loss(loss_dir)

        roi_name = getattr(ops, roi_config['type'])
        roi_config.pop('type')
        self.roi_layer = roi_name(**roi_config)

        self._init_layers()

    def _init_layers(self):
        self.cls_out_channels = self.num_classes

        self.conv_module = nn.Sequential(
            BasicBlock2d(self.input_channels, self.feat_channels),
            nn.Conv2d(self.input_channels, self.feat_channels, 3, padding=1))
        
        self.mlp_cls = self._mlp_module(self.feat_channels, self.num_classes)
        self.mlp_reg = self._mlp_module(self.feat_channels, self.box_code_size)
        self.mlp_orientation = self._mlp_module(self.feat_channels, 8)

    def _mlp_module(self, input_channels, out_channels):
        return nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels, out_channels),)

    def forward(self, features):
        features = self.conv_module(features)
        features = features.mean([2, 3])
        output = dict()
        output["cls"] = self.mlp_cls(features)
        output["reg"] = self.mlp_reg(features)
        output["orientation"] = self.mlp_orientation(features)
        return output

    def loss(self, refine_preds, gt, bbox_list, **kwargs):
        losses = {}
        mask = gt["cls"]!=-1
        reg_mask = (gt["cls"] != self.num_classes) & (gt["cls"] != -1)
        losses["loss_2stage_cls"] = self.loss_cls(refine_preds["cls"][mask], gt["cls"][mask],
                 torch.ones_like(gt["cls"][mask]), avg_factor=mask.sum())
        if reg_mask.sum() > 0:
            losses["loss_2stage_reg"] = self.loss_bbox(refine_preds["reg"][reg_mask], gt["reg"][reg_mask].detach(), 
                                torch.ones_like(gt["reg"][reg_mask]), avg_factor=reg_mask.sum())
            losses["2stage_reg_l1_distance"] = (refine_preds["reg"][reg_mask] - \
                                                    gt["reg"][reg_mask].detach()).abs().mean()

            if isinstance(self.loss_dir, dict) and self.loss_dir["type"] == "BinRotLoss":
                # bin_loss = BinRotLoss()
                pred_rot = refine_preds["orientation"][reg_mask]
                target_rot_bin = gt["rot_bin"][reg_mask]
                target_rot_res = gt["rot_res"][reg_mask]

                loss_orientation = compute_rot_loss(
                    pred_rot, target_rot_bin, target_rot_res, pred_rot.new_ones(len(pred_rot))  )
                
                losses["loss_2stage_orientation"] = loss_orientation
            losses["gt_refine_reg_max"] = gt["reg"][reg_mask].max()
        losses["pred_refine_reg_max"] = refine_preds["reg"][reg_mask].max()
        return losses    


    def select_features(self, bbox_list, features, img_metas, preds):
        # for pred, features, img_meta
        bboxes_2d_list = []
        if isinstance(features, list):
            features = features[-1]
        for idx in range(len(img_metas)):
            extrinsic = img_metas[idx]["lidar2cam"][0]
            extrinsic = torch.tensor(extrinsic).to(features.device)
            
            intrinsic = img_metas[idx]["cam2img"][0]
            intrinsic = torch.tensor(intrinsic).to(features.device)

            bboxes_3d = bbox_list[idx][0]
            bboxes_3d_cam = bboxes_3d.convert_to(Box3DMode.CAM, rt_mat=extrinsic)

            bboxes_2d = projected_2d_box(bboxes_3d_cam,
                rt_mat=intrinsic, img_shape=img_metas[idx]['img_shape'][0],)
            
            bboxes_2d = torch.cat(
                [bboxes_2d.new_ones(len(bboxes_2d), 1) * idx, bboxes_2d],
                dim=-1)
            
            bboxes_2d_list.append(bboxes_2d)
        bboxes_2d_list = torch.cat(bboxes_2d_list, dim=0)
        features_2d = self.roi_layer(features, bboxes_2d_list.detach())
        return features_2d
            
            
            