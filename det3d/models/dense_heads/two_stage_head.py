import numpy as np
import torch
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from torch import nn as nn

import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss

from mmdet3d.core.bbox.structures import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes
import torch.nn as nn
from .centernet_utils import CenterNetHeatMap, CenterNetDecoder, decode_dimension, gather_feature

# from inplace_abn import InPlaceABN
# from .loss_utils import reg_l1_loss, FocalLoss, depth_uncertainty_loss, BinRotLoss
import math
from mmdet3d.core.bbox.structures import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps

from .loss_utils import compute_rot_loss
from .centernet3d_head import get_orientation_bin, get_alpha
from det3d.core.bbox.util import projected_2d_box, alpha_to_ry, points_img2cam, points_img2cam_batch, bbox_alpha


@HEADS.register_module()

class TwoStageHead(nn.Module):
    """
    Two stage head for 3D detection models (especially for centernet).
    Given region of interest in 2D/3D space -> Generate 2D/3D voxel ->
        do the refinement.

    box_code_size: the num of variable for refining the bounding boxes
    refine_proposal: True -> do not modify the proposal, False: modify the proposal.
    """

    def __init__(self,  num_classes=3,
                        box_code_size=6, # x, y, z, dx, dy, dz
                        iou3d_thresh=0.4,
                        iou2d_thresh=0.75,
                        noise_scale=[1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2], # x, y, z dx, dy, dx, yaw
                        refine_proposal=True,
                        score_thresh=0.05,):
        super().__init__()
        self.num_classes = num_classes

        self.box_code_size = box_code_size
        self.iou3d_thresh = iou3d_thresh

        self.iou2d_thresh = iou3d_thresh
        # refine_proposal flase: keep the proposal as the same / else modify the proposal
        self.refine_proposal = refine_proposal

        noise_scale = torch.tensor(noise_scale).reshape(1, -1)
        self.register_buffer("noise_scale", noise_scale)

    def forward(self, features):
        raise NotImplementedError

    def simple_test(self, bbox_list, preds, img , img_metas, **kwargs):
        features = preds["features"]
        bbox_list = self.preprocess_prediction(bbox_list, img_metas, mode="inference")

        obj_features = self.select_features(bbox_list, features, img_metas, preds)

        refine_preds = self(obj_features)

        bbox_list = self.get_bbox_refine(refine_preds, bbox_list, img_metas)

        return bbox_list

    def get_bbox_refine(self, refine_preds, bbox_list, img_metas):

        bsz = len(img_metas)

        pred_cls = refine_preds["cls"].reshape(bsz, -1, self.num_classes)
        pred_reg = refine_preds["reg"].reshape(bsz, -1, self.box_code_size)
        num_pred = pred_reg.shape[1]
        pred_orientation = refine_preds["orientation"].reshape(bsz, num_pred, 8)
        new_bbox_list = []
        for idx in range(bsz):
            pred_bbox = bbox_list[idx][0]
            if self.refine_proposal is True:
                # check this which part ? ideally they should be in the camera coordinate
                extrinsic = img_metas[idx]["lidar2cam"][0]
                extrinsic = torch.tensor(extrinsic).to(pred_bbox.tensor.device)
                preds_bbox_cam = pred_bbox.convert_to(Box3DMode.CAM, rt_mat=extrinsic)

                alpha = get_alpha(pred_orientation[idx])

                rot_y = alpha_to_ry(preds_bbox_cam.center, alpha) #+ np.pi

                preds_bbox_cam.tensor[:,:self.box_code_size] += pred_reg[idx]

                pred_bbox = preds_bbox_cam.convert_to(Box3DMode.LIDAR, rt_mat=torch.inverse(extrinsic))

                # score, clses = pred_cls[idx].max(1)
                score, clses = bbox_list[idx][1], bbox_list[idx][2]
                # should filter the bboxes that score threshold small than xxx.

                new_bbox_list.append( (pred_bbox, score, clses))
            else:
                new_bbox_list.append( (bbox_list[idx]))

        return new_bbox_list


    def forward_train(self,
                    bbox_list,
                    preds,
                    img,
                    img_metas, gt_bboxes_3d,
                    gt_labels_3d, gt_bboxes):
        features = preds["features"]
        bbox_list = self.preprocess_prediction(bbox_list, img_metas, mode="training")
        bbox_list = self.insert_fake_preds_by_gt(bbox_list, img_metas, gt_bboxes_3d, gt_labels_3d)
        gt = self.generate_ground_truth(bbox_list, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes)

        obj_features = self.select_features(bbox_list, features, img_metas, preds)

        refine_preds = self(obj_features)
        losses = self.loss(refine_preds, gt, bbox_list)
        return losses, refine_preds

    def loss(self, refine_preds, gt, bbox_list, **kwargs):
        raise NotImplementedError

    def select_features(self, preds, img_metas):
        raise NotImplementedError

    def preprocess_prediction(self, preds, img_metas, mode="training"):
        if mode == "training":
            pass
        elif mode == "inference":
            pass
        return preds

    def insert_fake_preds_by_gt(self, preds, img_metas, gt_bboxes_3d, gt_labels_3d):
        if self.noise_scale.sum() == 0:
            return preds
        for idx in range(len(preds)):
            noise_gt = gt_bboxes_3d[idx].tensor.clone()
            noise = noise_gt.clone().uniform_(-1, 1)
            noise = noise * self.noise_scale
            noise_gt += noise
            preds[idx][0].tensor = torch.cat([preds[idx][0].tensor, noise_gt], dim=0)

            # 2. sample labels
            noise_score = preds[idx][1].new_ones(len(noise_gt))
            pred_scores = torch.cat([preds[idx][1], noise_score], dim=0)

            # 3. samples scores
            pred_labels = torch.cat([preds[idx][2], gt_labels_3d[idx].float()], dim=0)

            preds[idx] = (preds[idx][0], pred_scores, pred_labels)

        return preds

    def generate_ground_truth(self, preds, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes):
        # first check the gt -> 3d / 2d IoU ->
        # TODO for the missed gt -> #insert gt with random noise
        # TODO inject random noise in the depth dimension

        gt_cls = []
        gt_reg = []
        gt_rot_bin = []
        gt_rot_res = []
        gt_dict = {}


        for pred, img_meta, gt_bboxes_3d_idx, gt_labels_3d_idx, gt_bboxes_idx in zip(
                            preds, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes):

            gt_cls_idx, gt_reg_idx, gt_rot_bin_idx, gt_rot_res_idx = \
                self.generate_ground_truth_single(
                        pred, img_meta, gt_bboxes_3d_idx, gt_labels_3d_idx, gt_bboxes_idx)

            gt_cls.append(gt_cls_idx)
            gt_reg.append(gt_reg_idx)
            gt_rot_bin.append(gt_rot_bin_idx)
            gt_rot_res.append(gt_rot_res_idx)


        gt_dict["cls"] = torch.cat(gt_cls, dim=0)
        gt_dict["reg"] = torch.cat(gt_reg, dim=0)
        gt_dict["rot_bin"] = torch.cat(gt_rot_bin, dim=0)
        gt_dict["rot_res"] = torch.cat(gt_rot_res, dim=0)

        return gt_dict



    def generate_ground_truth_single(self, pred, img_meta, gt_bboxes_3d, gt_labels_3d, gt_bboxes):


        pred_bbox, pred_score, pred_class = pred
        iou_3d = pred_bbox.overlaps(pred_bbox, gt_bboxes_3d)
        device = gt_labels_3d[0].device

        intrinsic = torch.tensor(img_meta["cam2img"][0]).to(device)
        extrinsic = torch.tensor(img_meta["lidar2cam"][0]).to(device)

        pred_bbox_cam = pred_bbox.convert_to(Box3DMode.CAM, rt_mat = extrinsic)
        # pred_bbox_cam.tensor[:,6] -= np.pi
        pred_bbox_2d = projected_2d_box(pred_bbox_cam,
            rt_mat=intrinsic, img_shape=img_meta['img_shape'][0],)
        iou_2d = bbox_overlaps(pred_bbox_2d, gt_bboxes)

        iou_3d_pred_max, iou3d_pred_argmax = iou_3d.max(1)

        iou_2d_pred_max, iou2d_pred_argmax = iou_2d.max(1)

        pos_inds = iou3d_pred_argmax
        mask_2d = (iou_3d_pred_max<=self.iou3d_thresh) & (iou_2d_pred_max > self.iou3d_thresh)
        pos_inds[mask_2d] = iou2d_pred_argmax[mask_2d]

        neg_mask = (iou_3d_pred_max<=self.iou3d_thresh) & (iou_2d_pred_max <= self.iou3d_thresh)
        pos_inds[neg_mask] = -1

        gt_cls = gt_labels_3d[pos_inds]
        gt_cls[pos_inds==-1] = self.num_classes # TODO check the value for negatives

        # based on pos_inds -> generate ground truth

        # if (pos_inds!=-1).sum() > 0:
        gt_bboxes_3d_cam = gt_bboxes_3d.convert_to(Box3DMode.CAM, rt_mat = extrinsic)

        diff = gt_bboxes_3d_cam.tensor[pos_inds, :self.box_code_size] - pred_bbox_cam.tensor[:,:self.box_code_size]
        gt_reg = diff
        alpha = bbox_alpha(gt_bboxes_3d_cam)
        rot_bin, rot_res = get_orientation_bin(alpha[pos_inds])


        return gt_cls, gt_reg, rot_bin, rot_res

