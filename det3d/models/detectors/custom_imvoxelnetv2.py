import numpy as np
import os.path as osp
import os
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
from .custom_imvoxelnet import CustomImVoxelNet



@DETECTORS.register_module()
class CustomImVoxelNetv2(CustomImVoxelNet):
    r"""' Modify ImVoxelNet to satisfy current object detection modules."""
    def __init__(self,
                 backbone,
                 neck,
                 neck_3d=None,
                 view_transform=None,
                 neck_bev=None,
                 bbox_head=None,
                 n_voxels=None,
                 select_first_neck_feat=True,
                 bev_det_format=False,
                 train_cfg=None,
                 test_cfg=None,
                 video_mode = False,
                 aligned=False, # for bev4d
                 pre_process=None,
                 pre_process_neck=None,
                 before=False,
                 interpolation_mode="bilinear", # end for bev4d
                 with_bev_depth=False,
                 frame_num = 2,
                 with_prev=True,
                 video_aggregate_mode = "shift_feature",
                 pretrained=None,
                 init_cfg=None):
        BaseDetector.__init__(self, init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.n_voxels = n_voxels
        self.bev_det_format = bev_det_format
        self.with_bev_depth = with_bev_depth
        # if anchor_generator is not None:
        #     self.anchor_generator = build_prior_generator(anchor_generator)
        # else:
        #     self.anchor_generator = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.frame_num = frame_num
        self.with_prev = with_prev

        self.select_first_neck_feat = select_first_neck_feat
        if neck_3d is not None:
            self.neck_3d = build_neck(neck_3d)
        else:
            self.neck_3d = None
        if neck_bev is not None:
            self.neck_bev = build_neck(neck_bev)
        else:
            self.neck_bev = None
        if view_transform is not None:
            self.view_transform = build_neck(view_transform)
        else:
            self.view_transform = None
        self.video_mode = video_mode
        self.video_aggregate_mode = video_aggregate_mode
        self.aligned=aligned
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = build_neck(pre_process)
        self.pre_process_neck = pre_process_neck is not None
        if self.pre_process_neck:
            self.pre_process_neck_net = build_neck(pre_process_neck)
        self.before = before
        self.interpolation_mode = interpolation_mode
        self.distill = False


    def extract_bev_feat(self,
                         img_idx,
                         img_metas,
                         key_cam_matrix,
                         cam_matrix,
                         ref_cam_matrix,
                         frame_idx=0,
                         gt_depth=None):
        #input the camera matrics and output the bev feature
        x_fov = self.extract_img_feat(img_idx, img_metas)
        if isinstance(x_fov, list):
            x_fov = x_fov[0]
        _, feat_C, feat_H, feat_W = x_fov.shape
        bsz = cam_matrix[0].shape[0]
        x_fov = x_fov.reshape(bsz, -1, feat_C, feat_H, feat_W)
        if not self.with_bev_depth:
            x_3d, pred_depth = self.view_transform(
                [x_fov] + cam_matrix)
        else:
            x_3d, pred_depth = self.view_transform(
                [x_fov] + cam_matrix + [gt_depth])
        if self.video_aggregate_mode == "shift_feature":
            if self.before and self.pre_process:
                # pass to the preprocess_net before or after shift features
                x_3d = self.pre_process_net(x_3d)[0]
            if frame_idx != 0:
                x_3d = self.shift_feature(x_3d, key_cam_matrix, ref_cam_matrix)

            if self.pre_process and not self.before:
                x_3d = self.pre_process_net(x_3d)[0]


        return x_3d, x_fov, pred_depth


    @force_fp32()
    def shift_feature(self, input, key_cam_matrix, cam_matrix):
        #  trans, rots, ):
        n, c, h, w = input.shape
        _,v,_ =key_cam_matrix[1].shape

        # generate grid
        xs = torch.linspace(0, w - 1, w, dtype=input.dtype, device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=input.dtype, device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1).view(1, h, w, 3).expand(n, h, w, 3).view(n,h,w,3,1)
        grid = grid

        # get transformation from current frame to adjacent frame
        l02c = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        l02c[:,:,:3,:3] = key_cam_matrix[0]
        l02c[:,:,:3,3] = key_cam_matrix[1]
        l02c[:,:,3,3] =1

        l12c = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        l12c[:,:,:3,:3] = cam_matrix[0]
        l12c[:,:,:3,3] = cam_matrix[1]
        l12c[:,:,3,3] =1
        # l0tol1 = l12c.matmul(torch.inverse(l02c))[:,0,:,:].view(n,1,1,4,4)
        l0tol1 = l02c.matmul(torch.inverse(l12c))[:,0,:,:].view(n,1,1,4,4)
        l0tol1 = l0tol1[:,:,:,[True,True,False,True],:][:,:,:,:,[True,True,False,True]]
        feat2bev = torch.zeros((3,3),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.view_transform.dx[0]
        feat2bev[1, 1] = self.view_transform.dx[1]
        feat2bev[0, 2] = self.view_transform.bx[0] - self.view_transform.dx[0] / 2.
        feat2bev[1, 2] = self.view_transform.bx[1] - self.view_transform.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1,3,3)
        tf = torch.inverse(feat2bev).matmul(l0tol1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=input.dtype, device=input.device)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True, mode=self.interpolation_mode)
        return output



    def extract_video_feat(self, img, img_metas):
        img, aug_config = img[0], img[1:]
        B, N, _, H, W = img.shape
        N = N//self.frame_num
        img = img.view(B,N,self.frame_num,3,H,W)
        img = torch.split(img,1,2)
        img = [t.squeeze(2) for t in img]
        if not self.with_bev_depth:
            rots, trans, intrins, post_rots, post_trans = aug_config
            gt_depth = None
        else:
            rots, trans, intrins, post_rots, post_trans, gt_depth = aug_config
        extra = [rots.view(B,self.frame_num,N,3,3),
                 trans.view(B,self.frame_num,N,3),
                 intrins.view(B,self.frame_num,N,3,3),
                 post_rots.view(B,self.frame_num,N,3,3),
                 post_trans.view(B,self.frame_num,N,3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra

        x_3d_list = []
        x_fov_list = []

        for frame_idx, (img_idx, rot, tran, intrin, post_rot, post_tran) in enumerate(zip(img, rots, trans, intrins, post_rots, post_trans)):
            ref_cam_matrix = [rot, tran, intrin, post_rot, post_tran]
            if self.video_aggregate_mode == "shift_feature":
                tran = trans[0]
                rot = rots[0]
            cam_matrix = [rot, tran, intrin, post_rot, post_tran]
            if frame_idx == 0:
                key_cam_matrix = cam_matrix
            if frame_idx == 0 or self.with_prev:
                # Only calculate the temporal feature when with_prev=True
                if frame_idx == 0:
                    x_3d, x_fov, pred_depth = self.extract_bev_feat(
                                                                img_idx,
                                                                img_metas,
                                                                key_cam_matrix,
                                                                cam_matrix,
                                                                ref_cam_matrix,
                                                                frame_idx,
                                                                gt_depth)
                    x_fov_cur = x_fov              
                else:
                    with torch.no_grad():
                        x_3d, x_fov, _ = self.extract_bev_feat(
                                                            img_idx,
                                                            img_metas,
                                                            key_cam_matrix,
                                                            cam_matrix,
                                                            ref_cam_matrix,
                                                            frame_idx,
                                                            gt_depth)
            else:
                if frame_idx == 1:
                    x_3d = torch.zeros_like(x_3d_list[0])
                else:
                    x_3d = x_3d_list[-1]
            x_3d_list.append(x_3d)
        x_3d = torch.cat(x_3d_list, dim=1)
        if self.neck_3d is not None:
            x_bev = self.neck_3d(x_3d)
        else:
            x_bev = x_3d
        if self.neck_bev is not None:
            x_bev = self.neck_bev(x_bev)
        return x_3d, x_bev, x_fov_cur, pred_depth




@DETECTORS.register_module()
class FusionImVoxelNetv2(CustomImVoxelNetv2):
    '''
    The test-time augmentation version

    '''

    def __init__(self, save_path="./work_dirs",
                       load_path=["./work_dirs"],
                       inference_mode="save_results",
                       fusion_weights = [1., 1., 1.],
                       **kwargs):
        self.save_path = save_path
        self.load_path = load_path
        self.fusion_weights = np.array(fusion_weights)
        self.fusion_weights = self.fusion_weights /\
                                (np.sum(self.fusion_weights))
        self.inference_mode = inference_mode
        os.makedirs(self.save_path, exist_ok=True)

        super().__init__(**kwargs)
    def simple_test(self,
                    img_metas,
                    img=None,
                    img_inputs=None,
                    get_feats=None):
        
        if self.inference_mode == "save_results":
            return self.simple_test_inference(img_metas,
                                                img,
                                                img_inputs,
                                                get_feats)
        else:
            return self.simple_test_merge(img_metas,
                                            img,
                                            img_inputs,
                                            get_feats)            


    def simple_test_inference(self,
                    img_metas,
                    img=None,
                    img_inputs=None,
                    get_feats=None):
        if img is None:
            img = img_inputs
        if not self.video_mode:
            x_3d, x_bev, x_fov, pred_depth = self.extract_feat(img, img_metas)
        else:
            x_3d, x_bev, x_fov, pred_depth = self.extract_video_feat(img, img_metas)
        if isinstance(self.bbox_head, CenterHead):
            x_bev = [x_bev]
        x = self.bbox_head(x_bev)
        if not isinstance(self.bbox_head, CenterHead):
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas)
        else:
            bbox_list = self.bbox_head.get_bboxes(x, img_metas, rescale=False)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list]
        sample_idx = img_metas[0]['sample_idx']
        save_path = osp.join(self.save_path, sample_idx + "_inference.pth")
        torch.save(
                    dict(det_features=x,
                        det_results=bbox_results),
                    save_path)
        if get_feats is None or get_feats is False:
            return bbox_results
        elif get_feats == '3d':
            return bbox_results, x_3d
        elif get_feats == 'bev':
            return bbox_results, x_bev
        elif get_feats == 'fov':
            return bbox_results, x_fov


    def simple_test_merge(self,
                    img_metas,
                    img=None,
                    img_inputs=None,
                    get_feats=None):
        det_results_list = []
        det_features_list = []
        for load_path in self.load_path:
            sample_idx = img_metas[0]['sample_idx']
            load_path_idx = osp.join(load_path, sample_idx + "_inference.pth")
            temp = torch.load(load_path_idx)
            det_results_list.append(temp['det_results'])
            det_features_list.append(temp['det_features'])

        x = det_features_list[0]
        for idx, _ in enumerate(x):
            for key, item in x[idx][0].items():
                x[idx][0][key] = item * self.fusion_weights[0]
        
        for load_idx in range(1, len(self.load_path)):
            for idx, _ in enumerate(x):
                for key, item in x[idx][0].items():
                    temp = det_features_list[load_idx][idx][0][key]
                    x[idx][0][key] += temp * self.fusion_weights[load_idx]
        if not isinstance(self.bbox_head, CenterHead):
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas)
        else:
            bbox_list = self.bbox_head.get_bboxes(x, img_metas, rescale=False)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list]
        sample_idx = img_metas[0]['sample_idx']
        save_path = osp.join(self.save_path, sample_idx + "_inference.pth")
        torch.save(
                    dict(det_features=x,
                        det_results=bbox_results),
                    save_path)
        if get_feats is None or get_feats is False:
            return bbox_results
        elif get_feats == '3d':
            return bbox_results, x_3d
        elif get_feats == 'bev':
            return bbox_results, x_bev
        elif get_feats == 'fov':
            return bbox_results, x_fov


        raise NotImplementedError

