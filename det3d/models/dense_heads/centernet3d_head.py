import numpy as np
import torch
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from torch import nn as nn

import torch.nn.functional as F

from mmdet.models import HEADS
from mmdet3d.models.builder import build_loss
from mmdet3d.models.dense_heads.train_mixins import AnchorTrainMixin

from mmdet3d.core.bbox.structures import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes
import torch.nn as nn
from .centernet_utils import CenterNetHeatMap, CenterNetDecoder, decode_dimension, gather_feature


from inplace_abn import InPlaceABN
from .loss_utils import reg_l1_loss, FocalLoss, depth_uncertainty_loss, BinRotLoss
import math

from det3d.core.bbox.util import alpha_to_ry, points_img2cam, points_img2cam_batch, \
                    projected_gravity_center, projected_2d_box, bbox_alpha

def group_norm(out_channels):
    num_groups = 32
    if out_channels % 32 == 0:
        return nn.GroupNorm(num_groups, out_channels)
    else:
        return nn.GroupNorm(num_groups // 2, out_channels)


class SingleHead(nn.Module):
    def __init__(self, in_channel, conv_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, conv_channel, kernel_size=3, padding=1)
        # self.norm = group_norm(conv_channel)
        self.relu = InPlaceABN(conv_channel, momentum=0.1, activation="leaky_relu")
        self.out_conv = nn.Conv2d(conv_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)
        else:
            self.out_conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.feat_conv(x)
        # x = self.norm(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def get_alpha(rot):
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = torch.atan2(rot[:, 2], rot[:, 3]) + (-0.5 * math.pi)
    alpha2 = torch.atan2(rot[:, 6], rot[:, 7]) + (0.5 * math.pi)

    return alpha1 * idx + alpha2 * (~idx)


@HEADS.register_module()
class CenterNet3DHead(nn.Module, AnchorTrainMixin):
    """
    Args:
        head for centernet

    """

    def __init__(self,
                num_classes=3,
                input_channel=64,
                conv_channel=64,
                stride=4,
                tensor_dim=128,
                min_overlap=0.7,
                use_projected_box = True,
                depth_uncertainty_range = [-10, 10],
                dim_mean = ((3.8840, 1.5261, 1.6286),
                            (0.8423, 1.7606, 0.6602),
                            (1.7635, 1.7372, 0.5968)),
                dim_std = ((0.4529, 0.1367, 0.1022),
                            (0.2349, 0.1133, 0.1427),
                            (0.1766, 0.0948, 0.1242)),
                dim_mode =["exp", True, False],
                corner_loss_weight=0.2,
                inference_max_num=80,
                corner_origin=(0.5, 0.5, 0.5),
                depth_range=(1, 75),
                depth_uncertainty_as_conf=True,
                depth_uncertainty_as_conf_range=[0.01, 0.99],
                pred_dense_depth =False,
                calibrate_depth=False,
                dense_depth_cfg=dict(
                            supervised_mode="foreground",
                            threshold=0.1,
                            weight=0.1,),
                output_coordinate="lidar",
                bias_value=-2.19):
        # TODO fixed filter box with proejcted box
        super().__init__()
        self.cls_head = SingleHead(
            input_channel,
            conv_channel,
            num_classes,
            bias_fill=True,
            bias_value=bias_value)

        self.wh_head = SingleHead(input_channel, conv_channel, 2)
        self.reg_head = SingleHead(input_channel, conv_channel, 2)

        self.depth_head = SingleHead(input_channel, conv_channel, 1)

        self.dimension_head = SingleHead(input_channel, conv_channel, 3)
        self.orientation_head = SingleHead(input_channel, conv_channel, 8)
        self.amodel_offset_head = SingleHead(input_channel, conv_channel, 2)
        self.depth_uncertainty_head = SingleHead(input_channel, conv_channel, 1)
        torch.nn.init.xavier_normal_(self.depth_uncertainty_head.out_conv.weight, gain=0.1)

        self.num_classes = num_classes
        self.stride = stride
        self.tensor_dim = tensor_dim
        self.min_overlap = min_overlap
        self.use_projected_box = use_projected_box
        self.depth_uncertainty_range = depth_uncertainty_range
        self.dim_mode = dim_mode
        dim_mean = torch.Tensor(dim_mean)
        self.register_buffer("dim_mean", dim_mean)
        dim_std = torch.Tensor(dim_std)
        self.register_buffer("dim_std", dim_std)
        self.corner_loss_weight = float(corner_loss_weight)
        self.inference_max_num = inference_max_num
        self.corner_origin = corner_origin
        self.depth_range = depth_range
        self.depth_uncertainty_as_conf = depth_uncertainty_as_conf
        self.dense_depth_cfg = dense_depth_cfg
        self.depth_uncertainty_as_conf_range = depth_uncertainty_as_conf_range

        self.pred_dense_depth = pred_dense_depth
        self.output_coordinate = output_coordinate

        if pred_dense_depth:
            self.dense_depth_head = SingleHead(input_channel, conv_channel, 1)

        self.calibrate_depth = calibrate_depth


    def forward_train(self, x,
                         img_metas,
                         gt_bboxes_3d,
                         gt_labels_3d,
                         dense_depth=None,
                         **kwargs):
        preds = self(x)

        losses = self.loss(preds, img_metas, gt_bboxes_3d, gt_labels_3d, dense_depth, **kwargs)


        return losses, preds

    def forward_semi_train(self, x, img_metas, dense_depth=None, **kwargs):
        preds = self(x)
        losses = self.semi_loss(preds, img_metas, dense_depth, **kwargs)
        return losses, preds

    # @force_fp32(apply_to=(''))
    def loss(self, pred_dict, input_metas, gt_bboxes_3d,
                              gt_labels, dense_depth,
                              **kwargs):
        """

        """
        # generate ground truth
        bsz = len(input_metas)
        device = pred_dict["cls"].device

        intrinsic = torch.tensor([input_meta['cam2img'][0] for input_meta in input_metas])
        intrinsic = intrinsic.to(device)

        extrinsic = torch.tensor([input_meta['lidar2cam'][0] for input_meta in input_metas])
        extrinsic = extrinsic.to(device)

        output_shape = pred_dict["cls"].shape[2:]
        gt_dict = self.generate_ground_truth(input_metas,
                     gt_bboxes_3d, gt_labels, output_shape, intrinsic, extrinsic)
        # pass

        # for key, item in gt_dict.items():
            # gt_dict[key] = item.to(device)
        focal_loss = FocalLoss()
        loss_cls = focal_loss(pred_dict["cls"], gt_dict["scoremap"])
        mask = gt_dict["reg_mask"]
        index = gt_dict["index"]
        index = index.to(torch.long)

        loss_reg = reg_l1_loss(pred_dict["reg"], mask, index, gt_dict["reg"])

        loss_wh = reg_l1_loss(pred_dict["wh"], mask, index, gt_dict["wh"])

        # loss_depth_nouncertainty = reg_l1_loss(pred_dict["depth"], mask, index, gt_dict["depth"])
        pred_depth = pred_dict["depth"]
        pred_depth_uncertainty = pred_dict["depth_uncertainty"]
        pred_depth_uncertainty = torch.clamp(pred_depth_uncertainty,
                                            min = self.depth_uncertainty_range[0],
                                            max = self.depth_uncertainty_range[1])
        depth_l1_distance = reg_l1_loss(pred_depth, mask, index, gt_dict["depth"].unsqueeze(-1))
        loss_depth = depth_uncertainty_loss(pred_depth, pred_depth_uncertainty, mask, index,
                                            gt_dict["depth"].unsqueeze(-1), 1)

        pred_dimension_decode = decode_dimension(pred_dict["dimension"].clone(), gt_dict["scoremap"].clone(),
                                                self.dim_mean, self.dim_mode)

        pred_dict["dimension_decode"] = pred_dimension_decode

        loss_dimension = reg_l1_loss(pred_dict["dimension_decode"], mask, index, gt_dict["dimension"])
        loss_offset = reg_l1_loss(pred_dict["offset"], mask, index, gt_dict["offset"])
        if torch.isnan(loss_offset).sum() > 0:
            import pdb; pdb.set_trace()
        bin_loss = BinRotLoss()
        loss_orientation = bin_loss(pred_dict["orientation"], mask, index, gt_dict["rot_bin"], gt_dict["rot_res"])

        loss = {"loss_cls": loss_cls,
            "loss_box_wh": loss_wh,
            "loss_center_reg": loss_reg,
            "loss_depth": loss_depth,
            "loss_dimension": loss_dimension,
            "loss_offset": loss_offset,
            "loss_orientation": loss_orientation,
            "depth_l1_distance": depth_l1_distance,
            #"loss_corner": loss_corner,
            }
        # TODO add the corner loss
        # calculate the corner loss
        if self.corner_loss_weight > 0:
            gt_mask = gt_dict["gt_mask"]
            loss_corner = self.get_corner_loss(pred_dict, index, mask, gt_dict, gt_mask, gt_bboxes_3d, input_metas, intrinsic, extrinsic, output_shape)
            loss_corner = self.corner_loss_weight * loss_corner
            loss["loss_corner"] = loss_corner


        if self.pred_dense_depth:
            # two module 1. consider all objects
            # 2. only consider foreground
            # pred_bboxes = self.get_bboxes(pred_dict, input_metas)
            valid_mask, dense_depth_z = self.preprocess_dense_depth(
                                                        pred_dict,
                                                        input_metas,
                                                        dense_depth,
                                                        intrinsic,
                                                        extrinsic)
            if valid_mask.sum() == 0:
                loss["loss_dense_depth"] = loss_orientation.new_zeros(1)
            else:
                loss["loss_dense_depth"] = F.l1_loss(pred_dict["dense_depth"][valid_mask], dense_depth_z[valid_mask])
        # convert them to corner


        return loss


    def semi_loss(self, pred_dict, input_metas, dense_depth=None):
        """
        calculate the semi-supervised losss
        """
        device = pred_dict["cls"].device
        intrinsic = torch.tensor([input_meta['cam2img'][0] for input_meta in input_metas])
        intrinsic = intrinsic.to(device)

        extrinsic = torch.tensor([input_meta['lidar2cam'][0] for input_meta in input_metas])
        extrinsic = extrinsic.to(device)
        output_shape = pred_dict["cls"].shape[2:]
        loss = {}

        if self.pred_dense_depth:
            # two module 1. consider all objects
            # 2. only consider foreground
            # pred_bboxes = self.get_bboxes(pred_dict, input_metas)
            valid_mask, dense_depth_z = self.preprocess_dense_depth(
                                                        pred_dict,
                                                        input_metas,
                                                        dense_depth,
                                                        intrinsic,
                                                        extrinsic)
            if valid_mask.sum() == 0:
                loss["semi_loss_dense_depth"] = pred_dict["cls"].new_zeros(1)
            else:
                loss["semi_loss_dense_depth"] = F.l1_loss(pred_dict["dense_depth"][valid_mask], dense_depth_z[valid_mask])
        # convert them to corner
        return loss




    def get_corner_loss(self, pred_dict, index,
                         mask, gt_dict, gt_mask, gt_bboxes_3d,
                         input_metas, intrinsic, extrinsic, output_shape):
        # pass
        # 1. convert the prediction to 3D bounding box
        # index = index[mask]
        # batch_P =
        if mask.sum() == 0:
            return mask.new_zeros(1)
        bsz = index.shape[0]
        pred_u = index % output_shape[-1]
        pred_v = index // output_shape[-1]
        pred_offset = gather_feature(pred_dict["offset"], index, use_transform=True)
        pred_uv = torch.stack([pred_u, pred_v], dim=-1) + pred_offset
        pred_uv *= self.stride

        pred_depth = gather_feature(pred_dict["depth"], index, use_transform=True)
        if 'depth_ratio' in input_metas[0]:
            depth_ratio = torch.tensor([i['depth_ratio'] for i in input_metas])
            depth_ratio = depth_ratio.to(pred_depth.device).reshape(-1, 1, 1)
            pred_depth *= depth_ratio
        intrinsic = intrinsic.to(pred_uv.dtype)
        pred_location = points_img2cam_batch(pred_uv.reshape(-1,2),
                                       pred_depth.reshape(-1,1),
                                       intrinsic.unsqueeze(1).expand(-1,self.tensor_dim, -1, -1).reshape(-1, 4, 4))
        pred_location = pred_location.reshape(bsz, self.tensor_dim, 3)[mask.bool()]
        pred_dimension = gather_feature(pred_dict["dimension_decode"], index, use_transform=True)[mask.bool()]
        pred_orientation = gather_feature(pred_dict["orientation"], index, use_transform=True)[mask.bool()]
        pred_alpha = get_alpha(pred_orientation)
        pred_roty = alpha_to_ry(pred_location, pred_alpha).to(pred_dimension.dtype)
        pred_bboxes_3d = CameraInstance3DBoxes(torch.cat([
                                    pred_location.reshape(-1, 3),
                                    pred_dimension.reshape(-1, 3),
                                    pred_roty.reshape(-1,1)], dim=-1), origin=self.corner_origin)
        pred_bboxes_3d_corner = pred_bboxes_3d.corners
        gt_bboxes_3d_corner = torch.cat([
            gt_bbox_3d.convert_to(Box3DMode.CAM, extrinsic_idx).corners \
                 for gt_bbox_3d, extrinsic_idx in zip(gt_bboxes_3d, extrinsic)], dim=0)

        gt_bboxes_3d = torch.cat([
            gt_bbox_3d.convert_to(Box3DMode.CAM, extrinsic_idx).tensor \
                 for gt_bbox_3d, extrinsic_idx in zip(gt_bboxes_3d, extrinsic)], dim=0)

        gt_bboxes_3d_corner = gt_bboxes_3d_corner[gt_mask.bool()]
        corner_loss = F.l1_loss(pred_bboxes_3d_corner, gt_bboxes_3d_corner)
        return corner_loss
        # 2. covert the ground truth to 3D bounding box


    def preprocess_dense_depth(self, pred_dict, input_metas,
                                 dense_depth, intrinsic, extrinsic):
        # pred_bboxes = self.get_bboxes(pred_dict, input_metas)
        # inds =

        # get the filter mask by threshold

        # get the 2D projet boxes
        if self.dense_depth_cfg["supervised_mode"] == "foreground":
            pred_bboxes = self.get_bboxes(pred_dict, input_metas)

            fg_mask = torch.zeros_like(pred_dict["dense_depth"])

            pred_score = [i[1] for i in pred_bboxes]

            pred_bboxes_3d = [i[0] for i in pred_bboxes]

            # pred_bboxes_2d = []
            # modify it for each image
            for idx, pred_bbox_3d in enumerate(pred_bboxes_3d):

                pred_bbox_3d = pred_bbox_3d.convert_to(Box3DMode.CAM, rt_mat=extrinsic[idx])
                pred_bbox_2d = projected_2d_box(pred_bbox_3d,
                    rt_mat= intrinsic[idx], img_shape = input_metas[idx]['img_shape'][0])
                # pred_bboxes_2d.append(pred_bbox_2d)
                inds = pred_score[idx] > self.dense_depth_cfg['threshold']
                pred_bbox_2d = pred_bbox_2d[inds] / self.stride
                pred_bbox_2d = pred_bbox_2d.long()
                if len(pred_bbox_2d) > 0:
                    # do the for loop
                    for jdx in range(len(pred_bbox_2d)):
                        x1, y1, x2, y2 = pred_bbox_2d[jdx]
                        fg_mask[idx][:, y1:y2, x1:x2] = 1



                # pass
            # pred_bboxes_2d = [pred_bbox_3d.projected_2d_box(input_metas['intrinsic'], input_metas)]
            # pred_bboxes_2d =

            # downsample the dense_depth
            dense_depth_z = dense_depth[0][..., 2]
            dense_depth_z = dense_depth_z.unsqueeze(1)
            dense_depth_z = F.interpolate(dense_depth_z, pred_dict["dense_depth"].shape[-2:], mode="nearest")

            return fg_mask.bool(), dense_depth_z
        else:
            # fg_mask = torch.zeros_like(pred_dict["dense_ /depth"])
            gt_depth = dense_depth[0][..., 2]
            dense_depth_z = gt_depth.unsqueeze(1)
            dense_depth_z = F.interpolate(dense_depth_z, pred_dict["dense_depth"].shape[-2:], mode="nearest")
            return dense_depth_z > 0, dense_depth_z

    def get_bboxes(self, pred_dict, input_metas, cfg=None, recale=False, img=None):

        results = []
        box_scale = 1 / self.stride
        bsz = len(input_metas)
        for idx in range(bsz):
            cls_score = pred_dict["cls"][idx:idx+1]

            offset = pred_dict["offset"][idx:idx+1]
            depth = pred_dict["depth"][idx:idx+1]
            dimension = pred_dict["dimension"][idx:idx+1]
            orientation = pred_dict["orientation"][idx:idx+1]

            extrinsic = input_metas[idx]["lidar2cam"][0]
            intrinsic = input_metas[idx]["cam2img"][0]
            depth_uncertainty = pred_dict["depth_uncertainty"][idx:idx+1]



            if img is not None:
                img_idx = img[idx]
            else:
                img_idx = None

            if self.calibrate_depth is False:
                proposals = self.get_bboxes_single(cls_score, offset, depth, dimension, orientation, depth_uncertainty,
                                                extrinsic, intrinsic, box_scale = box_scale,
                                                input_meta=input_metas[idx], img_idx=img_idx)
            else:
                proposals = self.calibrate_bboxes_depth_single(cls_score, offset, depth, dimension,
                                                            orientation, depth_uncertainty,
                                                            extrinsic, intrinsic, box_scale=box_scale,
                                                            input_meta=input_metas[idx], img_idx=img_idx)
            results.append(proposals)

        return results
        # pass


    def get_bboxes_single(self, cls_scores, offset, depth,
                 dimension, orientation,  depth_uncertainty,
                 extrinsic, intrinsic, box_scale=1/4.,
                 input_meta=None, img_idx=None):
        K = self.inference_max_num
        batch, channel, height, width = cls_scores.shape
        cls_scores = CenterNetDecoder.pseudo_nms(cls_scores)

        scores, index, clses, ys, xs = CenterNetDecoder.topk_score(cls_scores, K=K)
        xs_int, ys_int = xs.clone(), ys.clone()

        clses = clses.reshape(1, K).float()

        scores = scores.reshape(1, K)


        dimension = decode_dimension(dimension, cls_scores, self.dim_mean, self.dim_mode)


        dimension = gather_feature(dimension, index, use_transform=True)

        dimension = dimension.reshape(K, 3)

        depth = gather_feature(depth, index, use_transform=True)

        depth = depth.reshape(K, 1)


        if self.depth_uncertainty_as_conf:
            depth_uncertainty = gather_feature(depth_uncertainty, index, use_transform=True)

            #depth_uncertainty = 1 - torch.clamp(depth_uncertainty.exp(), min=0.1, max=0.5)
            min_value, max_value = self.depth_uncertainty_as_conf_range
            depth_uncertainty = 1 - torch.clamp(depth_uncertainty.exp(), min=min_value, max=max_value)

            depth_uncertainty = depth_uncertainty.reshape(batch, K)
            #conf_mask = clses.cpu().apply_(lambda x: x in [2,3,4,5,7,9]).bool()

            #scores[conf_mask] = scores[conf_mask] * torch.sqrt(depth_uncertainty)[conf_mask]
            scores = scores * depth_uncertainty
        offset = gather_feature(offset, index, use_transform=True)

        offset = offset.reshape(K, 2)
        xs = xs_int.view(K, 1) + offset[:, 0:1]
        ys = ys_int.view(K, 1) + offset[:, 1:2]

        xs/=box_scale
        ys/=box_scale

        uv = torch.cat([xs, ys], dim=1)

        orientation = gather_feature(orientation, index, use_transform=True)
        orientation = orientation.reshape(K, 8)
        alpha = get_alpha(orientation)
        location = points_img2cam(uv, depth, torch.tensor(intrinsic).to(depth.device))


        rot_y = alpha_to_ry(location, alpha) # + np.pi
        location[:,1] += dimension[:,1]/2.
        # rot_y = limit_period(rot_y, 1, np.pi)
        boxes = torch.cat([location.reshape(K, -1), dimension.reshape(K, -1), rot_y.reshape(K, -1)], dim=-1)
        boxes = CameraInstance3DBoxes(boxes)

        

        if self.output_coordinate == "lidar":
            boxes = boxes.convert_to(Box3DMode.LIDAR,
                 rt_mat = torch.inverse(torch.tensor(extrinsic).cuda()))
        return boxes, scores.reshape(-1), clses.reshape(-1)



    def calibrate_bboxes_depth_single(self, cls_scores, offset, depth, dimension,
                                        orientation, depth_uncertainty, extrinsic, intrinsic, box_scale=4):

        K = self.inference_max_num
        batch, channel, height, width = cls_scores.shape
        cls_scores = CenterNetDecoder.pseudo_nms(cls_scores)

        scores, index, clses, ys, xs = CenterNetDecoder.topk_score(cls_scores, K=K)
        xs_int, ys_int = xs.clone(), ys.clone()

        clses = clses.reshape(1, K).float()

        scores = scores.reshape(1, K)


        dimension = decode_dimension(dimension, cls_scores, self.dim_mean, self.dim_mode)


        dimension = gather_feature(dimension, index, use_transform=True)

        dimension = dimension.reshape(K, 3)

        depth = gather_feature(depth, index, use_transform=True)

        depth = depth.reshape(K, 1)

        if self.depth_uncertainty_as_conf:
            depth_uncertainty = gather_feature(depth_uncertainty, index, use_transform=True)

            depth_uncertainty = 1 - torch.clamp(depth_uncertainty.exp(), min=0.01, max=0.99)

            depth_uncertainty = depth_uncertainty.reshape(batch, K)

            scores = scores * depth_uncertainty

        offset = gather_feature(offset, index, use_transform=True)

        offset = offset.reshape(K, 2)
        xs = xs_int.view(K, 1) + offset[:, 0:1]
        ys = ys_int.view(K, 1) + offset[:, 1:2]

        xs/=box_scale
        ys/=box_scale

        uv = torch.cat([xs, ys], dim=1)

        orientation = gather_feature(orientation, index, use_transform=True)
        orientation = orientation.reshape(K, 8)
        alpha = get_alpha(orientation)
        location = points_img2cam(uv, depth, torch.tensor(intrinsic).to(depth.device))


        rot_y = alpha_to_ry(location, alpha) # + np.pi
        location[:,1] += dimension[:,1]/2.
        # rot_y = limit_period(rot_y, 1, np.pi)
        boxes = torch.cat([location.reshape(K, -1), dimension.reshape(K, -1), rot_y.reshape(K, -1)], dim=-1)
        boxes = CameraInstance3DBoxes(boxes)
        boxes = boxes.convert_to(Box3DMode.LIDAR,
             rt_mat = torch.inverse(torch.tensor(extrinsic).cuda()))
        return boxes, scores.reshape(-1), clses.reshape(-1)



    def forward(self, x):
        if isinstance(x, list):
            multiscale_features = x
            x = x[-1]
        cls = self.cls_head(x)
        wh = self.wh_head(x)
        reg = self.reg_head(x)

        depth = self.depth_head(x)
        dimension = self.dimension_head(x)
        offset = self.amodel_offset_head(x)
        orientation = self.orientation_head(x)
        depth_uncertainty = self.depth_uncertainty_head(x)

        # process
        cls = _sigmoid(cls)
        depth = 1. / (depth.sigmoid() + 1e-6) - 1.
        # orientation convert
        vector_ori_bin1_channel = slice(2, 4)
        vector_ori_bin2_channel = slice(6, 8)
        vector_ori_bin1 = orientation[:, vector_ori_bin1_channel, ...].clone()
        orientation[:, vector_ori_bin1_channel, ...] = F.normalize(vector_ori_bin1)
        vector_ori_bin2 = orientation[:, vector_ori_bin2_channel, ...].clone()
        orientation[:, vector_ori_bin2_channel, ...] = F.normalize(vector_ori_bin2)

        pred = {"cls": cls, "wh": wh, "reg": reg,
                "depth": depth, "dimension": dimension,\
                "orientation": orientation, "offset": offset}
        pred["depth_uncertainty"] = depth_uncertainty

        if self.pred_dense_depth:
            pred["dense_depth"] = self.dense_depth_head(x)

        return pred

    # TODO fix the device from cpu to cuda
    def generate_ground_truth(self,
                            input_metas,
                            gt_bboxes_3d,
                            gt_labels,
                            output_shape,
                            intrinsic,
                            extrinsic,):
        box_scale = 1/float(self.stride)
        bsz = len(input_metas)
        scoremap_list, wh_list, reg_list, index_list, depth_list,\
             dimension_list, rot_bin_list, rot_res_list, alpha_list,\
             class_list, offset_list, reg_mask_list, gt_mask_list = [[] for i in range(13)]
        device = gt_labels[0].device
        # if use
        for idx in range(bsz):
            # TODO check output shape
            gt_scoremap = torch.zeros(self.num_classes, *output_shape, device=device)
            gt_wh = torch.zeros(self.tensor_dim, 2, device=device)
            gt_reg = torch.zeros_like(gt_wh)
            gt_index = torch.zeros(self.tensor_dim, device=device)
            gt_reg_mask = torch.zeros(self.tensor_dim, device=device)
            gt_classes = torch.zeros(self.tensor_dim, device=device)
            # add dense depth
            boxes_3d, classes = gt_bboxes_3d[idx], gt_labels[idx]
            num_bboxes = boxes_3d.tensor.shape[0]

            # in the camera_2 coordinate
            # convert to camera p0
            boxes_3d.tensor = boxes_3d.tensor.to(device)
            boxes_3d = Box3DMode.convert(
                boxes_3d, Box3DMode.LIDAR, Box3DMode.CAM, rt_mat = extrinsic[idx])
            if self.use_projected_box:
                boxes = projected_2d_box(boxes_3d,
                    rt_mat= intrinsic[idx], img_shape = input_metas[idx]['img_shape'][0])
            else:
                raise NotImplementedError
            if torch.isnan(boxes).sum() > 0:
                print(boxes, boxes_3d)
            centers = torch.stack([
                    boxes[:,[0,2]].mean(-1), boxes[:,[1,3]].mean(-1)], dim=-1)
            # print(centers)
            centers *= box_scale
            projected_centers = projected_gravity_center(boxes_3d, intrinsic[idx])

            projected_centers = projected_centers.to(device)
            projected_centers *= box_scale

            projected_centers_int = projected_centers.to(torch.int32)

            # centers_int = centers_int / 4
            mask = filter_out_img(projected_centers_int, input_metas[idx]['img_shape'][0], box_scale)
            mask = mask & filter_depth(boxes_3d, self.depth_range)
            # mask = mask & (boxes_3d.tensor[:,2] <= 1)
            mask = mask & (classes !=-1).bool().to(device)
            gt_mask_list.append(mask.reshape(-1))

            wh = torch.zeros_like(centers)

            wh[..., 0] = boxes[..., 2] - boxes[..., 0]
            wh[..., 1] = boxes[..., 3] - boxes[..., 1]
            wh *= box_scale
            if mask.sum() > 0:
                gt_scoremap_cpu = torch.zeros_like(gt_scoremap).to("cpu")
                CenterNetHeatMap.generate_scoremap(gt_scoremap, classes[mask], wh[mask], projected_centers_int[mask], min_overlap=self.min_overlap)

                # CenterNetHeatMap.generate_scoremap(gt_scoremap_cpu, classes[mask].to("cpu"), wh[mask].to("cpu"), projected_centers_int[mask].to("cpu"), min_overlap=self.min_overlap)
            gt_index[:num_bboxes] = projected_centers_int[..., 1] * output_shape[1] + projected_centers_int[..., 0]
            gt_reg[:num_bboxes] = centers - projected_centers_int

            gt_reg_mask[:num_bboxes] = mask



            gt_wh[:num_bboxes] = wh
            gt_classes[:num_bboxes] = classes


            scoremap_list.append(gt_scoremap)
            wh_list.append(gt_wh)
            reg_list.append(gt_reg)
            reg_mask_list.append(gt_reg_mask)
            class_list.append(gt_classes)
            index_list.append(gt_index)

            gt_depth = torch.zeros(self.tensor_dim, device=device)
            gt_dimension = torch.zeros(self.tensor_dim, 3, device=device)
            gt_offset = torch.zeros(self.tensor_dim, 2, device=device) # offset for float -> int

            gt_rot_bin = torch.zeros(self.tensor_dim, 2, device=device)
            gt_rot_res = torch.zeros(self.tensor_dim, 2, device=device)
            gt_alphas = torch.zeros(self.tensor_dim, device=device)
            gt_depth[:num_bboxes] = boxes_3d.tensor[:,2].to(device)
            if 'depth_ratio' in input_metas[idx]:
                assert input_metas[idx]['depth_ratio'] > 0
                gt_depth/= input_metas[idx]['depth_ratio']

            gt_dimension[:num_bboxes] = boxes_3d.tensor[:,3:6].to(device)
            gt_offset[:num_bboxes] = projected_centers - projected_centers_int
            if torch.isnan(gt_offset).sum() > 0 or torch.isinf(gt_offset).sum() > 0:
                import pdb; pdb.set_trace()
            alpha = bbox_alpha(boxes_3d).to(device)
            rot_bin, rot_res = get_orientation_bin(alpha)

            gt_rot_bin[:num_bboxes] = rot_bin
            gt_rot_res[:num_bboxes] = rot_res

            gt_alphas[:num_bboxes] = alpha


            depth_list.append(gt_depth)
            dimension_list.append(gt_dimension)
            offset_list.append(gt_offset)
            rot_bin_list.append(gt_rot_bin)
            rot_res_list.append(gt_rot_res)
            alpha_list.append(gt_alphas)
            # gt rot


        gt_dict = {
            "scoremap": torch.stack(scoremap_list, dim=0),
            "class": torch.stack(class_list, dim=0).unsqueeze(-1),
            "wh": torch.stack(wh_list, dim=0),
            "reg": torch.stack(reg_list, dim=0),
            "reg_mask": torch.stack(reg_mask_list, dim=0),
            "index": torch.stack(index_list, dim=0),
            "depth": torch.stack(depth_list, dim=0),
            "dimension": torch.stack(dimension_list, dim=0),
            "rot_bin": torch.stack(rot_bin_list, dim=0),
            "rot_res": torch.stack(rot_res_list, dim=0),
            "offset": torch.stack(offset_list, dim=0),
            "alpha": torch.stack(alpha_list, dim=0),
            "gt_mask": torch.cat(gt_mask_list, dim=0),
        }
        gt_dict["index"].clamp_(min=0, max=output_shape[0] * output_shape[1] - 1)

        return gt_dict


def filter_out_img(projected_box, input_shape, box_scale):
    input_shape = input_shape[:2]
    mask = (projected_box[:,0] >= 0) & (projected_box[:,1] >= 0) & \
            (projected_box[:,0] < input_shape[1] * box_scale) & \
            (projected_box[:,1] < input_shape[0] * box_scale)

    return mask



import math
def get_orientation_bin(alpha):
    """
    alpha: must between [-pi, pi], shape = [N, ]
    """
    bin0_mask = (alpha < math.pi / 6.) | (alpha > 5 * math.pi / 6.)
    bin1_mask = (alpha < - 5 * math.pi / 6.) | (alpha > - math.pi / 6.)

    rot_bin0 = bin0_mask.to(alpha.dtype)
    rot_res0 = alpha - (-0.5 * math.pi)
    rot_bin1 = bin1_mask.to(alpha.dtype)
    rot_res1 = alpha - (0.5 * math.pi)
    rot_bin = torch.stack([rot_bin0, rot_bin1], dim=1)
    rot_res = torch.stack([rot_res0 * rot_bin0, rot_res1 * rot_bin1], dim=1)
    return rot_bin, rot_res


def filter_depth(boxes, depth_range):
    depth = boxes.tensor[:, 2]
    mask = (depth > depth_range[0]) & (depth < depth_range[1])
    return mask


