from bdb import set_trace
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_backbone, build_head, build_neck
from mmdet.models.detectors import SingleStageDetector
import warnings
from mmdet3d.core import (Box3DMode, bbox3d2result,
                          LiDARInstance3DBoxes)
from mmdet3d.models.detectors import FCOSMono3D
from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
import copy

@DETECTORS.register_module()
class LocalVolume(FCOSMono3D):
    """
    Two stage pipeline for camera based 3d det
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 test_version="v1",
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)

        self.test_version = test_version
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def extract_img_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat(self, img):
        N, V, C, H, W = img.shape
        img = img.reshape(-1, C, H, W)
        return self.extract_img_feat(img)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      mono_gt_bboxes,
                      mono_gt_labels,
                      mono_gt_bboxes_3d,
                      mono_gt_labels_3d,
                      mono_centers2d,
                      mono_depths,
                      mono_attr_labels=None,
                      mono_gt_bboxes_ignore=None):
        # 1. handle the multiview images
        # check the img shape
        x = self.extract_feat(img)
        # 2. handle the gt bboxes;
        losses = {}
        losses_2d = self.get_bbox_head_losses(x, img_metas, mono_gt_bboxes,
                                                mono_gt_labels, mono_gt_bboxes_3d,
                                                mono_gt_labels_3d, mono_centers2d, mono_depths,
                                                mono_attr_labels, mono_gt_bboxes_ignore)
        for key, item in losses_2d.items():
            losses[key + "_2d_head"] = item

        return losses

    def get_bbox_head_losses(self, x, img_metas,  mono_gt_bboxes,
                                    mono_gt_labels, mono_gt_bboxes_3d,
                                    mono_gt_labels_3d, mono_centers2d, mono_depths,
                                    mono_attr_labels, mono_gt_bboxes_ignore):
        mono_gt_bboxes_list = []
        mono_gt_bboxes = [mono_gt_bboxes_list.extend(i) \
                                for i in mono_gt_bboxes][0]

        mono_gt_labels_list = []
        mono_gt_labels = [mono_gt_labels_list.extend(i) \
                                for i in mono_gt_labels][0]

        mono_gt_bboxes_3d_list = []
        mono_gt_bboxes_3d = [mono_gt_bboxes_3d_list.extend(i) \
                                for i in mono_gt_bboxes_3d][0]

        mono_gt_labels_3d_list = []
        mono_gt_labels_3d = [mono_gt_labels_3d_list.extend(i) \
                                for i in mono_gt_labels_3d][0]

        mono_centers2d_list = []
        mono_centers2d = [mono_centers2d_list.extend(i) \
                                for i in mono_centers2d][0]

        mono_depths_list = []
        mono_depths = [mono_depths_list.extend(i) \
                                for i in mono_depths][0]
        if mono_attr_labels is not None:
            mono_attr_labels_list = []
            mono_attr_labels = [mono_attr_labels_list.extend(i) \
                                    for i in mono_attr_labels][0]

        if mono_gt_bboxes_ignore is not None:
            mono_gt_bboxes_ignore_list = []
            mono_gt_bboxes_ignore = [mono_gt_bboxes_ignore_list.extend(i) \
                                        for i in mono_gt_bboxes_ignore][0]
        # check the useage of img_metas and the bboxes_3d;
        losses = self.bbox2d_head.forward_train(x, img_metas, mono_gt_bboxes,
                                                mono_gt_labels, mono_gt_bboxes_3d,
                                                mono_gt_labels_3d, mono_centers2d, mono_depths,
                                                mono_attr_labels, mono_gt_bboxes_ignore)
        return losses

    def forward_test(self, img, img_metas, rescale=False):
        return self.simple_test(img, img_metas, rescale)


    def simple_test(self, img, img_metas, rescale=False):
        # check how to combine the multi-view
        # N, V, C, H, W = img.shape
        # bbox_all = []
        # scores_all = []
        # labels_all = []
        # attrs_all = []
        # for vdx in range(V):
        #     x = self.extract_feat(img[:,vdx:vdx+1])
        #     outs = self.bbox_head(x)

        #     bbox_outputs_list = []
        #     view_idx = vdx % V

        #     # split the outs;
        #     img_metas_new = copy.deepcopy(img_metas)
        #     img_metas_new[0]['cam2img'] = img_metas_new[0]['cam2img'][view_idx]
        #     bbox_outputs = self.bbox_head.get_bboxes(
        #         *outs, img_metas_new, rescale=rescale)

        #     bbox, score, clses, attr = bbox_outputs[0]
        #     lidar2cam = img_metas[0]['lidar2cam'][view_idx]
        #     cam2lidar = score.new_tensor(np.linalg.inv(lidar2cam))
        #     bbox = bbox.convert_to(Box3DMode.LIDAR,
        #                         rt_mat=cam2lidar,
        #                         correct_yaw=True)
        #     bbox_all.append(bbox.tensor)
        #     scores_all.append(score)
        #     labels_all.append(clses)
        #     attrs_all.append(attr)

        
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_outputs_list = []
        img_metas_per_view = []
        N, V, C, H, W = img.shape
        for img_meta in img_metas:
            for view_idx in range(V):
                img_meta_idx = copy.deepcopy(img_meta)
                img_meta_idx['cam2img'] = img_meta_idx['cam2img'][view_idx]
                img_metas_per_view.append(img_meta_idx)

        bbox_outputs = self.bbox_head.get_bboxes(
            *outs, img_metas_per_view, rescale=rescale)
        bbox_all = []
        scores_all = []
        labels_all = []
        attrs_all = []
        for idx, img_meta in enumerate(img_metas_per_view):
            view_idx = idx % V
            bbox, score, clses, attr = bbox_outputs[idx]
            lidar2cam = img_meta['lidar2cam'][view_idx]
            cam2lidar = score.new_tensor(np.linalg.inv(lidar2cam))
            bbox = bbox.convert_to(Box3DMode.LIDAR,
                                   rt_mat=cam2lidar,
                                   correct_yaw=True)
            bbox_all.append(bbox.tensor)
            scores_all.append(score)
            labels_all.append(clses)
            attrs_all.append(attr)

        bbox_all = torch.cat(bbox_all, dim=0)
        scores_all = torch.cat(scores_all, dim=0)
        attrs_all = torch.cat(attrs_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        bbox_all = LiDARInstance3DBoxes(bbox_all, box_dim=9)

        nms_cfg = dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=4096,
            nms_thr=0.05,
            score_thr=0.01,
            min_bbox_size=0,
            max_per_frame=500)
        from mmcv import Config
        nms_cfg = Config(nms_cfg)

        bbox_all_for_nms = xywhr2xyxyr(bbox_all.bev)
        bbox_all_tensor = bbox_all.tensor
        nms_scores_all = scores_all.new_zeros(scores_all.shape[0], 10 + 1)
        indices = labels_all.new_tensor(list(range(scores_all.shape[0])))
        nms_scores_all[indices, labels_all] = scores_all
        bbox_all_tensor, scores_all, labels_all, attrs_all= box3d_multiclass_nms(
                        bbox_all_tensor,
                        bbox_all_for_nms,
                        nms_scores_all,
                        nms_cfg.score_thr,
                        nms_cfg.max_per_frame,
                        nms_cfg,
                        mlvl_attr_scores=attrs_all)
        topk = min(len(scores_all), 500)
        _, indices = torch.topk(scores_all, topk)
        bbox_all_tensor = bbox_all_tensor[indices]
        attrs_all = attrs_all[indices]
        labels_all = labels_all[indices]
        scores_all = scores_all[indices]
        bbox_all = LiDARInstance3DBoxes(bbox_all_tensor, box_dim=9)
        bbox_outputs = [(bbox_all, scores_all, labels_all, attrs_all)]
        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]
        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['pts_bbox'] = img_bbox

        return bbox_list


    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def show_results(self, img, img_metas, rescale=False):
        raise NotImplementedError
