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

from det3d.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class CustomImVoxelNet(BaseDetector):
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
                 detach=True,
                 test_adj_ids=None,
                 before=False,
                 interpolation_mode="bilinear", # end for bev4d
                 with_bev_depth=False,
                 frame_num = 2,
                 pretrained=None,
                 video_aggregate_mode = "shift_feature",
                 prev_no_grad=True,
                 init_cfg=None,
                 use_grid_mask=False):
        super().__init__(init_cfg)
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
        self.prev_no_grad = prev_no_grad

        self.grid_mask = GridMask(True,
                                  True,
                                  rotate=1,
                                  offset=False,
                                  ratio=0.5,
                                  mode=1,
                                  prob=0.7)
        self.use_grid_mask = use_grid_mask
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
        self.detach = detach
        self.test_adj_ids = test_adj_ids
        self.before = before
        self.interpolation_mode = interpolation_mode
        self.distill = False

    def extract_img_feat(self, img, img_metas):
        B = len(img)

        input_shape = img.shape[-2:]
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if img.dim() == 4:
            img = img.unsqueeze(0)
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        x_fov = self.backbone(img)
        if self.select_first_neck_feat:
            x_fov = self.neck(x_fov)[0]
        else:
            x_fov = self.neck(x_fov)
        return x_fov

    @force_fp32()
    def shift_feature(self, input, trans, rots, idx):
        n, c, h, w = input.shape
        _,v,_ =trans[0].shape

        # generate grid
        xs = torch.linspace(0, w - 1, w, dtype=input.dtype, device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=input.dtype, device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1).view(1, h, w, 3).expand(n, h, w, 3).view(n,h,w,3,1)
        grid = grid

        # get transformation from current frame to adjacent frame
        l02c = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        l02c[:,:,:3,:3] = rots[0]
        l02c[:,:,:3,3] = trans[0]
        l02c[:,:,3,3] =1

        l12c = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        l12c[:,:,:3,:3] = rots[idx]
        l12c[:,:,:3,3] = trans[idx]
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


    def extract_feat(self, img, img_metas):
        """Extract 3d features from the backboen -> fpn -> 3d projection.

        Args:
            img (torch.Tensor): Input images of shape (B, N, C_in, H, W)
            img_metas (list): Image metas.

        Returns:
            torch.Tensor: of shape (B, C_out, N_x, N_y, N_z)
        """
        # modify the shape
        if self.bev_det_format is True:
            img, aug_config = img[0], img[1:]

        B = len(img)
        x_fov = self.extract_img_feat(img, img_metas)
        if isinstance(x_fov, list):
            x_fov = x_fov[0]
        _, feat_C, feat_H, feat_W = x_fov.shape
        x_fov = x_fov.reshape(B, -1, feat_C, feat_H, feat_W)

        # if self.anchor_generator is not None:
        #     points = self.anchor_generator.grid_anchors(
        #         [self.n_voxels[::-1]], device=x_fov.device)[0][:, :3]
        # else:
        #     points = None
        if self.bev_det_format is False:
            x_3d = self.view_transform(x_fov, img_metas)
            pred_depth = None
        else:
            if self.with_bev_depth:
                x_3d, pred_depth = self.view_transform([x_fov] + aug_config)
            else:
                x_3d = self.view_transform([x_fov] + aug_config)
                pred_depth = None
        if self.neck_3d is not None:
            x_bev = self.neck_3d(x_3d)
        else:
            x_bev = x_3d

        if self.neck_bev is not None:
            x_bev = self.neck_bev(x_bev)
        return x_3d, x_bev, x_fov, pred_depth

    def extract_video_feat(self, img, img_metas):
        img, aug_config = img[0], img[1:]
        B, N, _, H, W = img.shape
        N = N//self.frame_num
        img = img.view(B,N,self.frame_num,3,H,W)
        img = torch.split(img,1,2)
        img = [t.squeeze(2) for t in img]
        if not self.with_bev_depth:
            rots, trans, intrins, post_rots, post_trans = aug_config
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
        bev_feat_list = []
        pred_depth_list = []
        for idx, (img_idx, rot, tran, intrin, post_rot, post_tran) in enumerate(zip(img, rots, trans, intrins, post_rots, post_trans)):
            if self.video_aggregate_mode == "shift_feature":
                tran = trans[0]
                rot = rots[0]
            if self.prev_no_grad and idx > 0:
                with torch.no_grad():
                    x_fov = self.extract_img_feat(img_idx, img_metas)
                    if isinstance(x_fov, list):
                        x_fov = x_fov[0]

                    _, feat_C, feat_H, feat_W = x_fov.shape
                    x_fov = x_fov.reshape(B, -1, feat_C, feat_H, feat_W)

                    if not self.with_bev_depth:
                        x_3d = self.view_transform(
                            [x_fov] + [rot, tran, intrin, post_rot, post_tran])
                        pred_depth = None
                    else:
                        x_3d, pred_depth = self.view_transform(
                            [x_fov] + [rot, tran, intrin, post_rot, post_tran, gt_depth])
            else:
                x_fov = self.extract_img_feat(img_idx, img_metas)
                if isinstance(x_fov, list):
                    x_fov = x_fov[0]

                _, feat_C, feat_H, feat_W = x_fov.shape
                x_fov = x_fov.reshape(B, -1, feat_C, feat_H, feat_W)

                if not self.with_bev_depth:
                    x_3d = self.view_transform(
                        [x_fov] + [rot, tran, intrin, post_rot, post_tran])
                    pred_depth = None
                else:
                    x_3d, pred_depth = self.view_transform(
                        [x_fov] + [rot, tran, intrin, post_rot, post_tran, gt_depth])

            bev_feat_list.append(x_3d)
            pred_depth_list.append(pred_depth)
        if self.neck_3d is not None:
            if self.video_aggregate_mode == "shift_feature":
                for idx in range(len(bev_feat_list)):
                    if idx > 0 and self.prev_no_grad is True:
                        bev_feat_list[idx] = self.neck_3d(bev_feat_list[idx])
                    else:
                        bev_feat_list[idx] = self.neck_3d(bev_feat_list[idx])
            else:
                if self.detach is True:
                    for idx in range(1, len(bev_feat_list)):
                        bev_feat_list[idx] = bev_feat_list[idx].detach()
                bev_feat_list = torch.cat(bev_feat_list, dim=1)
                bev_feat_list = self.neck_3d(bev_feat_list)
        if self.video_aggregate_mode == "shift_feature":
            if self.before and self.pre_process:
                for idx in range(len(bev_feat_list)):
                    if idx > 0 and self.prev_no_grad is True:
                        bev_feat_list[idx] = self.pre_process_net(bev_feat_list[idx])[0]
                    else:
                        bev_feat_list[idx] = self.pre_process_net(bev_feat_list[idx])[0]
            for idx in range(self.frame_num - 1):
                with torch.no_grad():
                    bev_feat_list[idx+1] = self.shift_feature(bev_feat_list[idx+1], trans, rots, idx+1)

            if self.pre_process and not self.before:
                if idx > 0 and self.prev_no_grad is True:
                    bev_feat_list[idx] = self.pre_process_net(bev_feat_list[idx])[0]
                else:
                    bev_feat_list[idx] = self.pre_process_net(bev_feat_list[idx])[0]
            x_bev = torch.cat(bev_feat_list, dim=1)
        else:
            x_bev = self.pre_process_net(bev_feat_list)[0]

        if self.neck_bev is not None:
            x_bev = self.neck_bev(x_bev)
        return _, x_bev, x_fov, pred_depth_list[0]




    @auto_fp16(apply_to=('img', 'img_inputs'))
    def forward(self, img_metas,
                      img=None,
                      img_inputs=None,
                      return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])
        if return_loss:
            return self.forward_train(img_metas,
                                      img=img,
                                      img_inputs=img_inputs,
                                      **kwargs)
        else:
            return self.forward_test(img_metas, img=img, img_inputs=img_inputs, **kwargs)


    def forward_test(self,
                    img_metas,
                    img=None,
                    img_inputs=None,
                    **kwargs):
        # currently the model do not support test time augmentation
        if not isinstance(img, list):
            img = [img]
            img_metas = [img_metas]
        return self.simple_test(img_metas[0], img[0], img_inputs=img_inputs)


    def forward_train(self, img_metas,
                      img=None,
                      img_inputs=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      return_output=False,
                      **kwargs):
        """Forward of training.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.

        Returns:
            dict[str, torch.Tensor]: A dictionary of loss components.
        """
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
            losses = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d,
                                         img_metas)
        else:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, x]
            losses = self.bbox_head.loss(*loss_inputs)

        if self.with_bev_depth:
            gt_depth = img[-1]
            if self.video_mode:
                B,N,H,W = gt_depth.shape
                gt_depth = torch.split(gt_depth.view(B,self.frame_num, N//self.frame_num, H, W), 1, 1, )[0].squeeze(1)
            loss_depth = self.view_transform.get_depth_loss(gt_depth, pred_depth)
            losses['loss_depth'] = loss_depth
        if return_output == False:
            return losses
        else:
            return losses, x_3d, x_bev, x_fov, x


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


    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            imgs (list[torch.Tensor]): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
