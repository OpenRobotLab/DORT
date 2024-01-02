import numpy as np
import torch
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

import torch.nn.functional as F
from mmdet3d.core import (PseudoSampler, box3d_multiclass_nms, limit_period,
                          xywhr2xyxyr)

from mmdet.models import HEADS
from mmdet.models.builder import HEADS, build_loss, build_neck

import torch.nn as nn

# from inplace_abn import InPlaceABN
# from .loss_utils import reg_l1_loss, FocalLoss, depth_uncertainty_loss, BinRotLoss
import math
# from mmdet3d.models.detectors.imvoxelnet import backproject, get_points

from .loss_utils import compute_rot_loss
from .two_stage_head import TwoStageHead

def generate_box_grid(n_voxels, origin="bottom_center", voxel_size=[8,8,6]):
    # generate grid
    # bottom center denotes the notation in 3D bbox
    x, y, z = n_voxels
    grid = torch.meshgrid([torch.arange(x), torch.arange(y), torch.arange(z)])

    grid = torch.stack(grid, dim=-1).float()
    grid[..., 0] *= voxel_size[0]
    grid[..., 1] *= voxel_size[1]
    grid[..., 2] *= voxel_size[2]
    grid[..., 0] -= (x-1) / 2 * (voxel_size[0])
    grid[..., 1] -= (y-1) / 2 * (voxel_size[1])
    grid[..., 2] -= (z-1) * voxel_size[2]
    grid[..., 2] *= -1
    # grid = torch.flip(grid, dims=[-2])
    return grid
@HEADS.register_module()
class BevObjectHead(TwoStageHead):
    """
    Two stage Bev object Head:
        Given region of interest in 3D space -> Generate 3D voxel ->
            two kinds of methods -> 1. with croping 2. with resizing
    """

    def __init__(self,
                neck_3d,
                num_classes=3,
                box_code_size=6, # x, y, z, w, h, l
                input_channels=64,
                feat_channels=64,
                decode_levels="1D",
                max_num_training=20,
                max_num_inference=50,
                n_voxels=[216, 256, 12],
                voxel_size=[0.32, 0.32, 0.32],
                select_feature_mode="crop",
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                loss_dir=dict(type='BinRotLoss'),
                iou3d_thresh=0.4,
                iou2d_thresh=0.75,
                insert_noise_with_gt=True,
                noise_scale=[1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2],
                refine_proposal=True,
                n_object_voxels=[16, 16, 12]):

        super().__init__(num_classes,
                        box_code_size,
                        iou3d_thresh,
                        iou2d_thresh,
                        noise_scale,
                        refine_proposal=refine_proposal)
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.select_feature_mode="crop"


        self.feat_channels = feat_channels
        self.decode_levels = decode_levels
        self.n_object_voxel = n_object_voxels

        self.neck_3d = build_neck(neck_3d)
        self.decode_levels = "1D"
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


        # for ground truth assignment


        obj_grid = generate_box_grid(n_object_voxels, "bottom_center", voxel_size)

        self.register_buffer("obj_grid", obj_grid)


        self._init_decoder_layers()


    def _init_decoder_layers(self):
        self.cls_out_channels = self.num_classes

        if self.decode_levels == "1D": # average pooling and then predict the box
            self.conv_cls = nn.Conv2d(self.feat_channels, self.feat_channels, 1)

            self.mlp_cls = _mlp_module(self.feat_channels, self.cls_out_channels)
            self.conv_reg = nn.Conv2d(self.feat_channels, self.feat_channels, 1)
            self.mlp_reg = _mlp_module(self.feat_channels, self.box_code_size)
            self.conv_orientation = nn.Conv2d(self.feat_channels, self.feat_channels, 1)
            self.mlp_orientation = _mlp_module(self.feat_channels, 8)

        elif self.decode_levels == "2D":
            # similar to anchor-free module that output the segmentation head.
            self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
            self.conv_reg = nn.Conv2d(self.feat_channels, self.box_code_size, 1)
            self.conv_orientation = nn.Conv2d(self.feat_channels, 8, 1)
            # self.conv_dir_cls = nn.Conv2d(self.feat_channels, 2, 1)



    def forward(self, features):
        if not isinstance(features, list):
            features = [features]
        features = self.neck_3d(features)
        features = features[0]
        output = {}

        if self.decode_levels == "1D":
            cls = self.conv_cls(features)
            cls = cls.mean([2,3])
            cls = self.mlp_cls(cls)
            output["cls"] = cls
            reg = self.conv_reg(features)
            reg = reg.mean([2,3])
            reg = self.mlp_reg(reg)
            output["reg"] = reg
            orientation = self.conv_orientation(features)
            orientation = orientation.mean([2,3])
            orientation = self.mlp_orientation(orientation)
            output["orientation"] = orientation
            return output

        elif self.decode_levels == "2D":
            cls = self.conv_cls(features)
            output["cls"] = cls
            reg = self.conv_reg(features)
            output["reg"] = reg
            direction = self.conv_dir(features)
            output["dir"] = direction
            return output



        # pass

        # 1. convert prediction to region of interest
        # 2. convert feature map to Voxel space
        # 3. different ways to select the feature map -> crop with background -> crop with foreground and resize
        # 4. rcnn head forward -> 3d conv -> 2d conv -> head with mlp? / or direct head
        # calculate loss -> adjustment loss or xxx

    def loss(self, refine_preds, gt, **kwargs):
        losses = {}
        mask = gt["cls"]!=-1
        reg_mask = (gt["cls"] != self.num_classes) & (gt["cls"] != -1)
        losses["loss_bev_cls"] = self.loss_cls(refine_preds["cls"][mask], gt["cls"][mask],
                 torch.ones_like(gt["cls"][mask]), avg_factor=mask.sum())
        if reg_mask.sum() > 0:
            losses["loss_bev_reg"] = self.loss_bbox(refine_preds["reg"][reg_mask], gt["reg"][reg_mask].detach(),
                                torch.ones_like(gt["reg"][reg_mask]), avg_factor=reg_mask.sum())

            if isinstance(self.loss_dir, dict) and self.loss_dir["type"] == "BinRotLoss":
                # bin_loss = BinRotLoss()
                pred_rot = refine_preds["orientation"][reg_mask]
                target_rot_bin = gt["rot_bin"][reg_mask]
                target_rot_res = gt["rot_res"][reg_mask]

                loss_orientation = compute_rot_loss(
                    pred_rot, target_rot_bin, target_rot_res, pred_rot.new_ones(len(pred_rot))  )

                losses["loss_bev_orientation"] = loss_orientation
            losses["gt_refine_reg_max"] = gt["reg"][reg_mask].max()
        losses["pred_refine_reg_max"] = refine_preds["reg"][reg_mask].max()
        return losses


    def generate_feature_voxel(self, features, img_metas):


        # 1. convert features to 3D space

        # 2. generate voxel grid for features

        # for features in [features]:
        stride = img_metas[0]['pad_shape'][-2] / features[0].shape[-1]
        stride = int(stride)

        assert stride == 4  # may be removed in the future
        volumes, valids = [], []
        features = features[0].unsqueeze(1)
        for idx, (feature, img_meta) in enumerate(zip(features, img_metas)):
            angles=None
            projection = self._compute_projection(img_meta, stride, angles).to(feature.device)
            points = get_points(
                n_voxels=torch.tensor(self.n_voxels),
                voxel_size=torch.tensor(self.voxel_size),
                origin=torch.tensor(img_meta['lidar2img']['origin'])
            ).to(feature.device)
            height = img_meta['img_shape'][0] // stride
            width = img_meta['img_shape'][1] // stride


            volume, valid = backproject(feature[:, :, :height, :width], points, projection)
            volume = volume.sum(dim=0)
            valid = valid.sum(dim=0)
            valid = valid > 0
            # volume[:, ~valid[0]] = .0
            volumes.append(volume)
            valids.append(valid)
        features_3d = torch.stack(volumes)
        valids = torch.stack(valids)
        # features_3d = [g]
        return features_3d

    def select_features(self, bbox_list, features,
                            img_metas, preds, **kwargs):
        '''
        Args:
        
        Output:
        
        '''
        
        features_3d = self.generate_feature_voxel(features, img_metas)
        # each grid for each features
        obj_features_3d = []
        for bbox, feature_3d, img_meta in zip(bbox_list, features_3d, img_metas):
            grid_index = self.generate_obj_index_grid(bbox, img_meta)
            grid_index = torch.flip(grid_index, dims=[-1])
            # flip because grid sample assume input with z y x
            grid_index = grid_index.detach()
            obj_features_3d.append(
                F.grid_sample(
                    feature_3d.unsqueeze(0).expand(grid_index.shape[0], -1, -1, -1, -1), grid_index,
                    mode="bilinear", padding_mode="zeros"))

        return torch.cat(obj_features_3d, dim=0)


    def generate_obj_index_grid(self, bbox, img_meta):

        # can be acclerated by putting it to gpu with sync
        # check the correctness of grid sample
        if self.select_feature_mode == "crop":
            # ignore obj dimension and obj shape
            # pred_bbox = [preds[0] for i in preds]
            pred_bbox = bbox[0]
            # obj_index_grid = []
            # for pred_bbox_idx in pred_bbox:
            box_centers = pred_bbox.center
            num_box = len(pred_bbox)
            obj_grid = self.obj_grid.unsqueeze(0).expand(num_box, -1, -1, -1, -1) + box_centers.reshape(-1, 1, 1, 1, 3)
            origin = img_meta['lidar2img']['origin']
            new_origin = torch.tensor(origin) - \
                    torch.tensor(self.n_voxels) / 2. * torch.tensor(self.voxel_size)

            obj_grid = obj_grid - new_origin.reshape(1,1,1,1,3).to(obj_grid.device)
            obj_grid = obj_grid / torch.tensor(self.n_voxels).reshape(1,1,1,1,3).float().to(obj_grid.device)
            obj_grid = obj_grid * 2 - 1
            # obj_grid = obj_grid /

            return obj_grid

        elif self.select_feature_mode == "align":
            import pdb; pdb.set_trace()

    # def generate_index_grid_singles


    @staticmethod
    def _compute_projection(img_meta, stride, angles):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio
        # use predicted pitch and roll for SUNRGBDTotal test
        if angles is not None:
            extrinsics = []
            for angle in angles:
                extrinsics.append(get_extrinsics(angle).to(intrinsic.device))
        else:
            extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
        for extrinsic in extrinsics:
            projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)





def _mlp_module(input_channels, out_channels):
    return nn.Sequential(
        nn.Linear(input_channels, input_channels),
        nn.BatchNorm1d(input_channels),
        nn.ReLU(inplace=True),
        nn.Linear(input_channels, out_channels),)