


import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.core import bbox3d2result
from mmdet3d.models.dense_heads import CenterHead
from mmdet3d.models.detectors import ImVoxelNet
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector
from mmcv.runner import auto_fp16
from mmcv.runner import force_fp32
import torch.nn as nn

from det3d.models.utils.grid_mask import GridMask
from .custom_imvoxelnet import CustomImVoxelNet

@DETECTORS.register_module()
class FusionImVoxelNet(CustomImVoxelNet):

    def __init__(self,
                num_model, 
                checkpoint_list = [],
                *args,
                **kwargs):
        BaseDetector.__init__(self)
        self.module_list = nn.ModuleList()

        for i in range(num_model):
            self.module_list.append(
                CustomImVoxelNet(*args, **kwargs))
        

        for i in range(num_model):
            ckpt = torch.load(checkpoint_list[i], map_location="cpu")
            self.module_list[i].load_state_dict(ckpt['state_dict'])

        self.num_model = num_model


    def simple_test(self, img_metas, img=None, img_inputs=None, get_feats=None):
        """Test without augmentations.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        if img is None:
            img = img_inputs
        x_list = []
        for i in range(self.num_model):
            if not self.module_list[0].video_mode:
                x_3d, x_bev, x_fov, pred_depth = self.module_list[i].extract_feat(img, img_metas)
            else:
                x_3d, x_bev, x_fov, pred_depth = self.module_list[i].extract_video_feat(img, img_metas)
            if isinstance(self.module_list[0].bbox_head, CenterHead):
                x_bev = [x_bev]
            x = self.module_list[i].bbox_head(x_bev)
            x_list.append(x)

        
        new_x = []
        for idx, result_idx in enumerate(x_list[0]):
            new_x.append([dict()])
            for key, item in result_idx[0].items():
                new_x[-1][0][key] = None
        
        for result in x_list[1:]:
            for idx, result_idx in enumerate(result):

                for key, item in result_idx[0].items():
                    if new_x[idx][0][key] is None:
                        new_x[idx][0][key] = item / float(self.num_model)
                    else:
                        new_x[idx][0][key] += item / float(self.num_model)

        x = new_x

        if not isinstance(self.module_list[0].bbox_head, CenterHead):
            bbox_list = self.module_list[0].bbox_head.get_bboxes(*x, img_metas)
        else:
            bbox_list = self.module_list[0].bbox_head.get_bboxes(x, img_metas, rescale=False)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        if get_feats is None or get_feats is False:
            return bbox_results
        elif get_feats == '3d':
            return bbox_results, x_3d
        elif get_feats == 'bev':
            return bbox_results, x_bev
        elif get_feats == 'fov':
            return bbox_results, x_fov