# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (get_proj_mat_by_coord_type,
                                          points_cam2img)
from det3d.core.bbox.util import custom_points_cam2img
from mmdet3d.models.fusion_layers import apply_3d_transformation


def custom_point_sample(img_meta,
                 img_features,
                 points,
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_pad_shape,
                 img_shape,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True,
                 valid_flag=False):
    """Obtain image features using points.

        !!! Modification:
            if img_features is type of list:,
             do the grid sample for the elements in the list
             and concat the output in the channel dimension

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)
    # project points to camera coordinate
    if valid_flag:
        proj_pts = custom_points_cam2img(points, proj_mat, with_depth=True)
        pts_2d = proj_pts[..., :2]
        depths = proj_pts[..., 2]
    else:
        pts_2d = points_cam2img(points, proj_mat)
    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1
    h, w = img_pad_shape

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    if not isinstance(img_features, list):
        point_features = F.grid_sample(
            img_features,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)  # 1xCx1xN feats
    else:
        point_features = []
        for img_feature in img_features:
            point_feature = F.grid_sample(
                img_feature,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners)
            point_features.append(point_feature)
        point_features = torch.cat(point_features, dim=1)


    if valid_flag:
        # (N, )
        valid = (coor_x.squeeze() < w) & (coor_x.squeeze() > 0) & (
            coor_y.squeeze() < h) & (coor_y.squeeze() > 0) & (
                depths > 0)
        valid_features = point_features.squeeze().t()
        valid_features[~valid] = 0
        return valid_features, valid  # (N, C), (N,)

    return point_features.squeeze().t()



def lss_custom_point_sample(img_meta,
                            img_features,
                            points,
                            proj_mat,
                            coord_type,
                            img_scale_factor,
                            img_crop_offset,
                            img_flip,
                            img_pad_shape,
                            img_shape,
                            aligned=True,
                            padding_mode='zeros',
                            align_corners=True,
                            valid_flag=False,
                            depth_range=[0, 50]):
    """Obtain image features using points.

        !!! Modification:
            if img_features is type of list:,
             do the grid sample for the elements in the list
             and concat the output in the channel dimension

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    # project points to camera coordinate
    if valid_flag:
        proj_pts = custom_points_cam2img(points, proj_mat, with_depth=True)
        pts_2d = proj_pts[..., :2]
        depths = proj_pts[..., 2]
    else:
        pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1
    coor_z = depths
    h, w = img_pad_shape

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    coor_z = (coor_z - depth_range[0]) / (depth_range[1] - depth_range[0])
    norm_coor_z = coor_z * 2 - 1
    norm_coor_z = norm_coor_z.unsqueeze(1)
    grid = torch.cat([norm_coor_x, norm_coor_y, norm_coor_z],
                     dim=1).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    if not isinstance(img_features, list):
        point_features = F.grid_sample(
            img_features,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)  # 1xCx1xN feats
    else:
        point_features = []
        for img_feature in img_features:
            point_feature = F.grid_sample(
                img_feature,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners)
            point_features.append(point_feature)
        point_features = torch.cat(point_features, dim=1)


    if valid_flag:
        # (N, )
        valid = (coor_x.squeeze() < w) & (coor_x.squeeze() > 0) & (
            coor_y.squeeze() < h) & (coor_y.squeeze() > 0) & (
                depths > 0) & (depths < depth_range[1])
        valid_features = point_features.squeeze().t()
        valid_features[~valid] = 0
        return valid_features, valid  # (N, C), (N,)

    return point_features.squeeze().t()




def custom_stereo_point_sample(img_meta,
                            img_features,
                            points,
                            proj_mat,
                            coord_type,
                            img_scale_factor,
                            img_crop_offset,
                            img_flip,
                            img_pad_shape,
                            img_shape,
                            aligned=True,
                            padding_mode='zeros',
                            depth_range=[2, 59.6],
                            depth_size=144,
                            align_corners=True):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    # project points to camera coordinate
    pts_2d = points_cam2img(points, proj_mat, with_depth=True)
    pts_2d, depth = pts_2d[..., :2], pts_2d[...,2:3]

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1
    h, w = img_pad_shape
    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        pad_gap = max(h - orig_h, 0)

        coor_x = orig_w - coor_x + pad_gap
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1
    coor_z = depth.clone() - depth_range[0]
    coor_z = (coor_z / depth_size) *2 - 1
    grid_3d = torch.cat([coor_z, coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2
    # align_corner=True provides higher performance
    mask = (depth > 0) & \
            (coor_y >= 0) & (coor_y <=1) &\
            (coor_x >= 0) & (coor_y <= 1)
    return grid_3d, mask
