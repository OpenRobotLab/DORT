
import torch
import numpy as np
from mmdet3d.core import bbox3d2result, build_prior_generator

from det3d.models.fusion_layers.point_fusion import custom_point_sample
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector
from mmdet3d.models.detectors import ImVoxelNet
from det3d.models.utils.grid_mask import GridMask
from mmdet3d.models.dense_heads import CenterHead
from det3d.models.fusion_layers.point_fusion import custom_stereo_point_sample
import torch.nn.functional as F


def project_pseudo_lidar_to_rectcam(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([-ys, -zs, xs], dim=-1)


def project_rectcam_to_pseudo_lidar(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([zs, -xs, -ys], dim=-1)


def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n, 1), device=pts_3d_rect.device)
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

@DETECTORS.register_module()
class LigaStereo(BaseDetector):
    def __init__(self,
                backbone,
                neck_stereo,
                build_cost_volume,
                neck_cost_volume,
                neck_voxel,
                neck_voxel_to_bev,
                neck_bev,
                depth_head,
                det3d_head,
                maxdisp=144,
                downsampled_disp=4,
                downsampled_depth_offset=0.2,
                point_cloud_range=[2, -30.4, -3, 59.6, 30.4, 1],
                voxel_size = [0.2, 0.2, 0.2],
                neck_det2d=None,
                det2d_head=None,
                cat_img_feature=True,
                img_feature_attentionbydisp = True,
                voxel_attentionbydisp = False,
                use_stereo_out_type="feature",
                grid_mask=False,
                with_upconv=True,
                init_cfg=None,
                pretrained=None,
                train_cfg=None,
                test_cfg=None):
        super().__init__(init_cfg)
        point_cloud_range = np.array(point_cloud_range)
        voxel_size = np.array(voxel_size)

        grid_size = (
            point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)

        self.backbone = build_backbone(backbone)
        self.neck_stereo = build_neck(neck_stereo)  # concat the backbone features

        self.neck_cost_volume = build_neck(neck_cost_volume)
        self.build_cost_volume = build_neck(build_cost_volume)

        self.depth_head = build_head(depth_head)

        # for the 3d part
        self.neck_voxel = build_neck(neck_voxel)
        self.neck_voxel_to_bev = build_neck(neck_voxel_to_bev)
        self.neck_bev = build_neck(neck_bev)
        det3d_head.update(train_cfg=train_cfg)
        det3d_head.update(test_cfg=test_cfg)
        self.det3d_head = build_head(det3d_head)


        self.maxdisp = maxdisp
        self.downsampled_disp = downsampled_disp
        self.downsampled_depth_offset = downsampled_depth_offset
        self.grid_mask = grid_mask
        self.fullres_stereo_feature = with_upconv
        self.cat_img_feature = cat_img_feature

        # self.neck_cost_volume = build_neck()

        if det2d_head is not None:
            self.det2d_head = build_head(det2d_head) # ATSS head
        if neck_det2d is not None:
            self.neck_det2d = build_neck(neck_det2d) # FPN neck


        self.img_feature_attentionbydisp = img_feature_attentionbydisp
        self.voxel_attentionbydisp = voxel_attentionbydisp
        self.use_stereo_out_type = use_stereo_out_type

        self.prepare_depth(point_cloud_range, in_camera_view=False)
        self.prepare_coordinates_3d(point_cloud_range, voxel_size, grid_size)

    def prepare_depth(self, point_cloud_range, in_camera_view=False):
        if in_camera_view:
            self.CV_DEPTH_MIN = point_cloud_range[2]
            self.CV_DEPTH_MAX = point_cloud_range[5]
        else:
            self.CV_DEPTH_MIN = point_cloud_range[0]
            self.CV_DEPTH_MAX = point_cloud_range[3]
        assert self.CV_DEPTH_MIN >= 0 and self.CV_DEPTH_MAX > self.CV_DEPTH_MIN
        depth_interval = (self.CV_DEPTH_MAX - self.CV_DEPTH_MIN) / self.maxdisp
        print('stereo volume depth range: {} -> {}, interval {}'.format(self.CV_DEPTH_MIN,
                                                                        self.CV_DEPTH_MAX, depth_interval))
        # prepare downsampled depth
        self.downsampled_depth = torch.zeros(
            (self.maxdisp // self.downsampled_disp), dtype=torch.float32)
        for i in range(self.maxdisp // self.downsampled_disp):
            self.downsampled_depth[i] = (
                i + self.downsampled_depth_offset) * self.downsampled_disp * depth_interval + self.CV_DEPTH_MIN
        # prepare depth
        self.depth = torch.zeros((self.maxdisp), dtype=torch.float32)
        for i in range(self.maxdisp):
            self.depth[i] = (
                i + 0.5) * depth_interval + self.CV_DEPTH_MIN
    def prepare_coordinates_3d(self, point_cloud_range, voxel_size, grid_size, sample_rate=(1, 1, 1)):
        self.X_MIN, self.Y_MIN, self.Z_MIN = point_cloud_range[:3]
        self.X_MAX, self.Y_MAX, self.Z_MAX = point_cloud_range[3:]
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = voxel_size
        self.GRID_X_SIZE, self.GRID_Y_SIZE, self.GRID_Z_SIZE = grid_size.tolist()

        self.VOXEL_X_SIZE /= sample_rate[0]
        self.VOXEL_Y_SIZE /= sample_rate[1]
        self.VOXEL_Z_SIZE /= sample_rate[2]

        self.GRID_X_SIZE *= sample_rate[0]
        self.GRID_Y_SIZE *= sample_rate[1]
        self.GRID_Z_SIZE *= sample_rate[2]

        zs = torch.linspace(self.Z_MIN + self.VOXEL_Z_SIZE / 2., self.Z_MAX - self.VOXEL_Z_SIZE / 2.,
                            self.GRID_Z_SIZE, dtype=torch.float32)
        ys = torch.linspace(self.Y_MIN + self.VOXEL_Y_SIZE / 2., self.Y_MAX - self.VOXEL_Y_SIZE / 2.,
                            self.GRID_Y_SIZE, dtype=torch.float32)
        xs = torch.linspace(self.X_MIN + self.VOXEL_X_SIZE / 2., self.X_MAX - self.VOXEL_X_SIZE / 2.,
                            self.GRID_X_SIZE, dtype=torch.float32)
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coordinates_3d = torch.stack([xs, ys, zs], dim=-1)
        self.coordinates_3d = coordinates_3d.float()

    @torch.no_grad()
    def get_local_depth(self, d_prob):
        d = self.depth.cuda()[None, :, None, None]
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

    def get_stereo_baseline(self, img_metas):
        stereo_baseline = [
                img_meta['cam2img'][0][0,3] - img_meta['cam2img'][1][0,3] for img_meta in img_metas]
        stereo_baseline = torch.tensor(stereo_baseline)
        return stereo_baseline
    def extract_feat(self, img, img_metas, mode):

        output = {}
        B = img.size(0)
        input_shape = img.shape[-2:]


        fu_mul_baseline = self.get_stereo_baseline(img_metas).to(img.device)

        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if img.dim() == 4:
            img = img.unsqueeze(0)
        B, N, C, H, W = img.size()
        # img = img.view(B*N, C, H, W)
        assert N == 2
        if self.grid_mask:
            img = self.grid_mask(img)
        left = img[:, 0]
        right = img[:, 1]
        # img_feats = self.backbone(img)
        left_img_feats = self.backbone(img[:,0])
        right_img_feats = self.backbone(img[:,1])

        left_img_feats = [img[:,0]] + list(left_img_feats)
        right_img_feats = [img[:,1]] + list(right_img_feats)
        left_stereo_feats, left_det_feats = self.neck_stereo(left_img_feats)
        right_stereo_feats, right_det_feats = self.neck_stereo(right_img_feats)


        if self.neck_det2d is not None:
            output["left_det_feats"] = self.neck_det2d([left_det_feats])
        else:
            output["left_det_feats"] = [left_det_feats]

        left_rpn_feats = left_det_feats

        output["det_feats"] = left_det_feats
        output["rpn_feats"] = left_det_feats


        downsampled_depth = self.downsampled_depth.cuda()

        downsampled_disp = fu_mul_baseline[:, None] / \
            downsampled_depth[None, :] / (self.downsampled_disp.cuda() if not self.fullres_stereo_feature else 1)


        cost_raw = self.build_cost_volume(left_stereo_feats, right_stereo_feats, None, None, downsampled_disp)

        all_costs = self.neck_cost_volume(cost_raw)

        # img_feats = img_feats.reshape(B, N, C, )
        # left_feats, right_feats =
        # left_stereo_fea


        depth_preds, depth_volumes, depth_cost_softmax, depth_preds_local = self.depth_head(all_costs, self.depth.cuda(), left.shape[-2:])
        output["depth_preds"] = depth_preds
        output["depth_volumes"] = depth_volumes
        output["depth_cost_softmax"] = depth_cost_softmax
        output["depth_preds_local"] = depth_preds_local


        if self.use_stereo_out_type == "feature":
            out = all_costs[-1]
        elif self.use_stereo_out_type == "prob":
            out = depth_cost_softmax[-1]
        elif self.use_stereo_out_type == "cost":
            out = depth_volumes[-1]
        else:
            raise ValueError('wrong self.use_stereo_out_type option')

        out_prob = depth_cost_softmax[-1]

        # do the feature conversion
        norm_coord_imgs, coord_imgs, valids_2d, valids = self.get_norm_coord_imgs(img_metas)
        voxel = F.grid_sample(out, norm_coord_imgs, align_corners=True)
        voxel = voxel * valids[:, None, :, :, :]
        if (self.voxel_attentionbydisp or
                (self.img_feature_attentionbydisp and self.cat_img_feature)):
            pred_disp = F.grid_sample(out_prob.detach()[:, None],
                                      norm_coord_imgs, align_corners=True)
            pred_disp = pred_disp * valids[:, None, :, :, :]

            if self.voxel_attentionbydisp:
                voxel = voxel * pred_disp

        if self.cat_img_feature:
            rpn_feature = left_det_feats
            norm_coord_imgs_2d = norm_coord_imgs.clone().detach()
            norm_coord_imgs_2d[..., 2] = 0
            voxel_2d = F.grid_sample(rpn_feature.unsqueeze(2), norm_coord_imgs_2d, align_corners=True)
            voxel_2d = voxel_2d * valids_2d.float()[:, None, :, :, :]
            if self.img_feature_attentionbydisp:
                voxel_2d = voxel_2d * pred_disp

            if voxel is not None:
                voxel = torch.cat([voxel, voxel_2d], dim=1)
            else:
                voxel = voxel_2d

        voxel_nopool, voxel = self.neck_voxel(voxel)

        output["voxel_nopool_features"] = voxel_nopool
        output["voxel_features"] = voxel
        return output

    def get_norm_coord_imgs(self, img_metas):
        # modify current voxel sample module
        c3d = self.coordinates_3d.view(-1, 3).cuda()
        c3d_extend = torch.cat(
            [c3d, c3d.new_ones(len(c3d), 1)], dim=-1)

        norm_coord_imgs = []
        coord_imgs = []
        valids = []
        valids_2d = []
        # valids =
        for i in range(len(img_metas)):
            img_meta = img_metas[i]
            lidar2img = img_meta["lidar2img"]
            coord_img = torch.matmul(c3d_extend, c3d_extend.new_tensor(lidar2img[0]).cuda().t())
            img_shape = img_meta["img_shape"][0]

            coord_img = coord_img[:,:3]
            coord_img[:,:2] /= (coord_img[:,2:3].abs() + 1e-4)
            # norm_coord_imgs
            coord_imgs.append(coord_img.view(*self.coordinates_3d.shape[:3], 3))
            valid = (coord_img[..., 0] >= 0) \
                                & (coord_img[..., 0]<= img_shape[1]) \
                                & (coord_img[..., 1]>=0) \
                                & (coord_img[..., 1]<=img_shape[0]) 

            valids_2d.append(valid.view(*self.coordinates_3d.shape[:3]))
            valid = valid & (coord_img[..., 2]>= self.CV_DEPTH_MIN) \
                                & (coord_img[..., 2]<= self.CV_DEPTH_MAX)
            valids.append(valid.view(*self.coordinates_3d.shape[:3]))
            coord_x, coord_y, coord_z = torch.split(coord_img, 1, dim=1)

            coord_x /= img_shape[1]
            coord_y /= img_shape[0]
            coord_z -= self.CV_DEPTH_MIN
            coord_z /= (self.CV_DEPTH_MAX - self.CV_DEPTH_MIN)
            norm_coord_img = torch.cat([coord_x, coord_y, coord_z], dim=1)
            norm_coord_img = norm_coord_img * 2 - 1
            norm_coord_imgs.append(norm_coord_img.view(*self.coordinates_3d.shape[:3], 3))
        norm_coord_imgs = torch.stack(norm_coord_imgs, dim=0)
        coord_imgs = torch.stack(coord_imgs, dim=0)
        valids_2d = torch.stack(valids_2d, dim=0)
        valids = torch.stack(valids, dim=0)
        return norm_coord_imgs, coord_imgs, valids_2d, valids





    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):

        output = self.extract_feat(img, img_metas, "train")
        output = self.neck_voxel_to_bev(output)
        output = self.neck_bev(output)
        det_output = self.det3d_head([output["spatial_features_2d"].permute(0,1,3,2)])

        losses = self.det3d_head.loss(*det_output, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def forward_dummy(self, img):

        if img.dim() == 4: # fix the situation of multiview inputs
            img = img.unsqueeze(0)
        x = self.extract_feat(img, None, 'test')
        outs = self.bbox_head.forward(x)
        return outs


    def forward_test(self, img, img_metas, **kwargs):
        if not isinstance(img, list):
            img = [img]
            img_metas = [img_metas]

        return self.simple_test(img[0], img_metas[0], **kwargs)
        # aug test is under developed
        #else:
        #    return self.aug_test(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        #N, V, C, H, W = img.shape
        output = self.extract_feat(img, img_metas, "val")
        output = self.neck_voxel_to_bev(output)
        output = self.neck_bev(output)

        det_output = self.det3d_head([output["spatial_features_2d"].permute(0,1,3,2)])

        bbox_list = self.det3d_head.get_bboxes(*det_output, img_metas, rescale=False)

        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img, img_metas, rescale=False):
        raise NotImplementedError

