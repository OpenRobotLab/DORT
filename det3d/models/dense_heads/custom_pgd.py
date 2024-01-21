# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


from mmdet3d.models.builder import HEADS
from mmdet3d.models.dense_heads import PGDHead
from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr
from mmdet3d.core.bbox import points_img2cam
from mmdet.core import distance2bbox
from det3d.core.bbox.util import centernet_distance2bbox
from mmdet3d.core import CameraInstance3DBoxes
INF = 1e8

@HEADS.register_module()
class CustomPGDHead(PGDHead):
    '''Customize modification of PGD
    Args:
        centernet_dim (bool, optional): Whether to regress the
            object dim as centernet and do not regress the offset
            between point with bbox boundary(fcos).
    '''

    def __init__(self,
                 centernet_dim=False,
                 inference_bbox2d=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.centernet_dim = centernet_dim
        self.inference_bbox2d = inference_bbox2d

    # def

    def _get_target_single(self, gt_bboxes, gt_labels, gt_bboxes_3d,
                           gt_labels_3d, centers2d, depths, attr_labels,
                           points, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels_3d.new_full(
                       (num_points,), self.background_label), \
                   gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)), \
                   gt_bboxes_3d.new_zeros((num_points,)), \
                   attr_labels.new_full(
                       (num_points,), self.attr_background_label)

        # change orientation to local yaw
        gt_bboxes_3d[..., 6] = -torch.atan2(
            gt_bboxes_3d[..., 0], gt_bboxes_3d[..., 2]) + gt_bboxes_3d[..., 6]

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers2d = centers2d[None].expand(num_points, num_gts, 2)
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, num_gts,
                                                 self.bbox_code_size)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        delta_xs = (xs - centers2d[..., 0])[..., None]
        delta_ys = (ys - centers2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat(
            (delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1)
        # add the if
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if self.centernet_dim:
            center_x = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2.
            delta_2d_xs = (xs - center_x)
            center_y = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2.
            delta_2d_ys = (ys - center_y)
            dim_x = gt_bboxes[..., 2] - gt_bboxes[..., 0]
            dim_y = gt_bboxes[..., 3] - gt_bboxes[..., 1]
            bbox_targets_center = torch.stack(
                        (delta_2d_xs, delta_2d_ys, dim_x, dim_y), -1)

            # bbox_targets_centernet =
        assert self.center_sampling is True, 'Setting center_sampling to '\
            'False has not been implemented for FCOS3D.'
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        center_xs = centers2d[..., 0]
        center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # center-based criterion to deal with ambiguity
        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1))
        dists[inside_gt_bbox_mask == 0] = INF
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(dim=1)

        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        attr_labels = attr_labels[min_dist_inds]
        labels[min_dist == INF] = self.background_label  # set as BG
        labels_3d[min_dist == INF] = self.background_label  # set as BG
        attr_labels[min_dist == INF] = self.attr_background_label

        bbox_targets = bbox_targets[range(num_points), min_dist_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]
        if self.centernet_dim:
            bbox_targets_center = bbox_targets_center[range(num_points), min_dist_inds]
            bbox_targets = bbox_targets_center
        relative_dists = torch.sqrt(
            torch.sum(bbox_targets_3d[..., :2]**2,
                      dim=-1)) / (1.414 * stride[:, 0])
        # [N, 1] / [N, 1]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)

        return labels, bbox_targets, labels_3d, bbox_targets_3d, \
            centerness_targets, attr_labels



    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           depth_cls_preds,
                           weights,
                           attr_preds,
                           centernesses,
                           mlvl_points,
                           input_meta,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W).
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on a single scale level with shape
                (num_points * 2, H, W)
            depth_cls_preds (list[Tensor]): Box scores for probabilistic depth
                predictions on a single scale level with shape
                (num_points * self.num_depth_cls, H, W)
            weights (list[Tensor]): Location-aware weight maps on a single
                scale level with shape (num_points * self.weight_dim, H, W).
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            input_meta (dict): Metadata of input image.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            tuples[Tensor]: Predicted 3D boxes, scores, labels, attributes and
                2D boxes (if necessary).
        """
        view_list = np.array(input_meta['cam2img'])
        scale_factor_list = input_meta['scale_factor']

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []
        mlvl_depth_cls_scores = []
        mlvl_depth_uncertainty = []
        mlvl_bboxes2d = None
        if self.pred_bbox2d:
            mlvl_bboxes2d = []

        for idx, (cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
                attr_pred, centerness, points) in enumerate(zip(
                    cls_scores, bbox_preds, dir_cls_preds, depth_cls_preds,
                    weights, attr_preds, centernesses, mlvl_points)):
            if isinstance(scale_factor_list, list):
                scale_factor = scale_factor_list[idx]
            else:
                scale_factor = scale_factor_list
            if len(view_list.shape) == 3:
                view = view_list[idx]
            else:
                view = view_list

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            depth_cls_pred = depth_cls_pred.permute(1, 2, 0).reshape(
                -1, self.num_depth_cls)
            depth_cls_score = F.softmax(
                depth_cls_pred, dim=-1).topk(
                    k=2, dim=-1)[0].mean(dim=-1)
            if self.weight_dim != -1:
                weight = weight.permute(1, 2, 0).reshape(-1, self.weight_dim)
            else:
                weight = weight.permute(1, 2, 0).reshape(-1, 1)
            depth_uncertainty = torch.exp(-weight[:, -1])
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred3d = bbox_pred[:, :self.bbox_coder.bbox_code_size]
            if self.pred_bbox2d:
                bbox_pred2d = bbox_pred[:, -4:]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                merged_scores = scores * centerness[:, None]
                if self.use_depth_classifier:
                    merged_scores *= depth_cls_score[:, None]
                    if self.weight_dim != -1:
                        merged_scores *= depth_uncertainty[:, None]
                max_scores, _ = merged_scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred3d = bbox_pred3d[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                depth_cls_pred = depth_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                depth_cls_score = depth_cls_score[topk_inds]
                depth_uncertainty = depth_uncertainty[topk_inds]
                attr_score = attr_score[topk_inds]
                if self.pred_bbox2d:
                    bbox_pred2d = bbox_pred2d[topk_inds, :]
            # change the offset to actual center predictions
            bbox_pred3d[:, :2] = points - bbox_pred3d[:, :2]
            if rescale:
                #if isinstance(scale_factor, float):
                #    bbox_pred3d[:, :2] /= bbox_pred3d[:, :2].new_tensor(
                #        scale_factor)
                #else:
                #    bbox_pred3d[:, :2] /= bbox_pred3d[:, :2].new_tensor(
                #        scale_factor[:2])
                if self.pred_bbox2d:
                    if not self.centernet_dim:
                        bbox_pred2d /= bbox_pred2d.new_tensor(scale_factor)
                    else:
                        bbox_pred2d[..., :2] /= bbox_pred2d[...,:2].new_tensor(scale_factor)
            if self.use_depth_classifier:
                prob_depth_pred = self.bbox_coder.decode_prob_depth(
                    depth_cls_pred, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda)
                bbox_pred3d[:, 2] = sig_alpha * bbox_pred3d[:, 2] + \
                    (1 - sig_alpha) * prob_depth_pred
            pred_center2d = bbox_pred3d[:, :3].clone()
            bbox_pred3d[:, :3] = points_img2cam(bbox_pred3d[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred3d)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_depth_cls_scores.append(depth_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
            mlvl_depth_uncertainty.append(depth_uncertainty)
            if self.pred_bbox2d:
                img_shape = input_meta['img_shape']
                if isinstance(img_shape, list):
                    img_shape = img_shape[idx]
                if not self.centernet_dim:
                    bbox_pred2d = distance2bbox(
                        points, bbox_pred2d, max_shape=img_shape)
                else:
                    bbox_pred2d = centernet_distance2bbox(
                        points, bbox_pred2d, max_shape=img_shape)
                mlvl_bboxes2d.append(bbox_pred2d)
        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        if self.pred_bbox2d:
            mlvl_bboxes2d = torch.cat(mlvl_bboxes2d)

        # change local yaw to global yaw for 3D nms
        cam2img = torch.eye(
            4, dtype=mlvl_centers2d.dtype, device=mlvl_centers2d.device)
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)
        mlvl_bboxes_for_nms = xywhr2xyxyr(CameraInstance3DBoxes(
            mlvl_bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        # no scale_factors in box3d_multiclass_nms
        # Then we multiply it from outside
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        if self.use_depth_classifier:  # multiply the depth confidence
            mlvl_depth_cls_scores = torch.cat(mlvl_depth_cls_scores)
            mlvl_nms_scores *= mlvl_depth_cls_scores[:, None]
            if self.weight_dim != -1:
                mlvl_depth_uncertainty = torch.cat(mlvl_depth_uncertainty)
                mlvl_nms_scores *= mlvl_depth_uncertainty[:, None]
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg.score_thr,
                                       cfg.max_per_img, cfg, mlvl_dir_scores,
                                       mlvl_attr_scores, mlvl_bboxes2d)
        bboxes, scores, labels, dir_scores, attrs = results[0:5]
        attrs = attrs.to(labels.dtype)  # change data type to int
        bboxes = CameraInstance3DBoxes(
            bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5))
        # Note that the predictions use origin (0.5, 0.5, 0.5)
        # Due to the ground truth centers2d are the gravity center of objects
        # v0.10.0 fix inplace operation to the input tensor of cam_box3d
        # So here we also need to add origin=(0.5, 0.5, 0.5)
        if not self.pred_attrs:
            attrs = None

        outputs = (bboxes, scores, labels, attrs)
        if self.pred_bbox2d:
            bboxes2d = results[-1]
            bboxes2d = torch.cat([bboxes2d, scores[:, None]], dim=1)
            if self.inference_bbox2d:
                outputs = outputs + (bboxes2d, )
        return outputs
