import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.core import build_prior_generator

from mmdet.models import NECKS
import numpy as np

from det3d.models.fusion_layers.point_fusion import custom_point_sample, \
                                                    lss_custom_point_sample

@NECKS.register_module()
class ImVoxelViewTransform(nn.Module):
    def __init__(self,
                n_voxels,
                anchor_generator,
                valid_sample=True):
        super().__init__()
        self.n_voxels = n_voxels
        self.valid_sample = valid_sample
        self.anchor_generator = build_prior_generator(anchor_generator)

    def forward(self, x_fov, img_metas):

        points = self.anchor_generator.grid_anchors(
             [self.n_voxels[::-1]], device=x_fov.device)[0][:, :3]
        volumes = []
        img_scale_factors = []
        img_flips = []
        img_crop_offsets = []
        for batch_idx, (feature, img_meta) in enumerate(zip(x_fov, img_metas)):
            if 'scale_factor' in img_meta:
                if isinstance(
                        img_meta['scale_factor'],
                        np.ndarray) and len(img_meta['scale_factor']) >= 2:
                    img_scale_factor = (points.new_tensor(
                        img_meta['scale_factor'][:2]))
                else:
                    img_scale_factor = (points.new_tensor(
                        img_meta['scale_factor']))
            else:
                img_scale_factor = (1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (points.new_tensor(img_meta['img_crop_offset'])
                               if 'img_crop_offset' in img_meta.keys() else 0)
            img_scale_factors.append(img_scale_factor)
            img_flips.append(img_flip)
            img_crop_offsets.append(img_crop_offset)

            volume_idx = []
            mask_idx = []
            for jdx in range(feature.size(0)):

                img_shape = img_meta['img_shape'][jdx][:2]
                sample_results = custom_point_sample(
                    img_meta,
                    img_features=feature[jdx][None, ...],
                    points=points,
                    proj_mat=points.new_tensor(img_meta['ori_lidar2img'][jdx]),
                    coord_type='LIDAR',
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img_meta['input_shape'],
                    img_shape=img_shape,
                    aligned=False,
                    valid_flag=self.valid_sample)

                if self.valid_sample:
                    volume_idx.append(sample_results[0])
                    mask_idx.append(sample_results[1])
                else:
                    volume_idx.append(sample_results)

            if not self.valid_sample:
                volume = torch.stack(volume_idx, dim=0).mean(0)
            else:
                valid_nums = torch.stack(mask_idx,
                                         dim=0).sum(0).float()  # (N, )
                volume = torch.stack(volume_idx, dim=0).sum(0)
                valid_mask = valid_nums > 0
                volume[~valid_mask] = 0
                volume = volume / valid_nums[:, None].clamp(min=1e-3)
            import pdb; pdb.set_trace()
            volumes.append(
                volume.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0))
        x_3d = torch.stack(volumes)
        return x_3d





@NECKS.register_module()
class LSSImVoxelViewTransform(nn.Module):
    def __init__(self,
                n_voxels,
                anchor_generator,
                valid_sample=True,
                input_channel=64,
                depth_range = [0, 50],
                depth_grid_size=0.75):
        super().__init__()
        self.n_voxels = n_voxels
        self.valid_sample = valid_sample
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.depth_grid_num = int((depth_range[1] - depth_range[0]) / depth_grid_size)
        self.depth_range = depth_range
        self.depth_net = nn.Conv2d(input_channel, input_channel + self.depth_grid_num, kernel_size=1, padding=0)


    def forward(self, x_fov, img_metas):
        B, N, C, H, W = x_fov.shape

        x_fov = self.depth_net(x_fov.reshape(B*N, C, H, W))

        depth, x_fov = x_fov[:,:self.depth_grid_num], x_fov[:,self.depth_grid_num:]

        depth = depth.softmax(dim=1)
        D = depth.shape[1]

        points = self.anchor_generator.grid_anchors(
             [self.n_voxels[::-1]], device=x_fov.device)[0][:, :3]

        x_fov = depth.unsqueeze(1) * x_fov.unsqueeze(2)
        x_fov = x_fov.reshape(B, N, C, D, H, W)



        volumes = []
        img_scale_factors = []
        img_flips = []
        img_crop_offsets = []
        for batch_idx, (feature, img_meta) in enumerate(zip(x_fov, img_metas)):
            if 'scale_factor' in img_meta:
                if isinstance(
                        img_meta['scale_factor'],
                        np.ndarray) and len(img_meta['scale_factor']) >= 2:
                    img_scale_factor = (points.new_tensor(
                        img_meta['scale_factor'][:2]))
                else:
                    img_scale_factor = (points.new_tensor(
                        img_meta['scale_factor']))
            else:
                img_scale_factor = (1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (points.new_tensor(img_meta['img_crop_offset'])
                               if 'img_crop_offset' in img_meta.keys() else 0)
            img_scale_factors.append(img_scale_factor)
            img_flips.append(img_flip)
            img_crop_offsets.append(img_crop_offset)

            volume_idx = []
            mask_idx = []
            for jdx in range(feature.size(0)):

                img_shape = img_meta['img_shape'][jdx][:2]

                sample_results = lss_custom_point_sample(
                    img_meta,
                    img_features=feature[jdx][None, ...],
                    points=points,
                    proj_mat=points.new_tensor(img_meta['ori_lidar2img'][jdx]),
                    coord_type='LIDAR',
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img_meta['input_shape'],
                    img_shape=img_shape,
                    aligned=False,
                    valid_flag=self.valid_sample,
                    depth_range=self.depth_range)

                if self.valid_sample:
                    volume_idx.append(sample_results[0])
                    mask_idx.append(sample_results[1])
                else:
                    volume_idx.append(sample_results)

            if not self.valid_sample:
                volume = torch.stack(volume_idx, dim=0).mean(0)
            else:
                valid_nums = torch.stack(mask_idx,
                                         dim=0).sum(0).float()  # (N, )
                volume = torch.stack(volume_idx, dim=0).sum(0)
                valid_mask = valid_nums > 0
                volume[~valid_mask] = 0
                volume = volume / valid_nums[:, None].clamp(min=1e-3)
            volumes.append(
                volume.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0))
        x_3d = torch.stack(volumes)
        return x_3d
