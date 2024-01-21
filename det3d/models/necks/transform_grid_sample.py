# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from mmdet3d.models import builder
from mmcv.cnn import build_conv_layer
import torch.nn.functional as F
from .lift_splat import SELikeModule
from mmdet3d.core import build_prior_generator
from .lift_splat import  gen_dx_bx

@NECKS.register_module()
class ViewTransformerGridSample(BaseModule):
    def __init__(self,
                 n_voxels,
                 anchor_generator,
                 extra_depth_net,
                 loss_depth_weight,
                 se_config=dict(),
                 dcn_config=dict(bias=True),
                 grid_config=None,
                 data_config=None,
                 numC_input=512,
                 numC_Trans=64,
                 downsample=16,
                 bev_pool=True):
        

        super(ViewTransformerGridSample, self).__init__()
        self.grid_config = grid_config
        self.data_config = data_config
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.downsample = downsample
        self.bev_pool = bev_pool
        
        self.D = grid_config['dbound'][1] - grid_config['dbound'][0]
        self.D = int(self.D / grid_config['dbound'][2])
        # 1. build the points_generator
        self.n_voxels = n_voxels
        self.anchor_generator = build_prior_generator(anchor_generator)
        points = self.anchor_generator.grid_anchors(
            [self.n_voxels[::-1]])[0][:,:3]
        self.register_buffer("points", points)
        # 2. build the depth net

        self.loss_depth_weight = loss_depth_weight
        self.extra_depthnet = builder.build_backbone(extra_depth_net)
        self.featnet = nn.Conv2d(self.numC_input,
                                 self.numC_Trans,
                                 kernel_size=1,
                                 padding=0)
        self.depthnet = nn.Conv2d(extra_depth_net['num_channels'][0],
                                  self.D,
                                  kernel_size=1,
                                  padding=0)
        self.dcn = nn.Sequential(*[build_conv_layer(dict(type='DCNv2',
                                                        deform_groups=1),
                                                   extra_depth_net['num_channels'][0],
                                                   extra_depth_net['num_channels'][0],
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   dilation=1,
                                                   **dcn_config),
                                   nn.BatchNorm2d(extra_depth_net['num_channels'][0])
                                  ])
        self.se = SELikeModule(self.numC_input,
                               feat_channel=extra_depth_net['num_channels'][0],
                               **se_config)

        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        self.fH = fH
        self.fW = fW
        self.ogfH = ogfH
        self.ogfW = ogfW
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.register_buffer("dx", dx)
        self.register_buffer("bx", bx)
        self.register_buffer("nx", nx)

    def forward(self, input):
        if len(input) == 6:
            x, rots, trans, intrins, post_rots, post_trans = input
        else:
            x, rots, trans, intrins, post_rots, post_trans, depth_gt = input

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        img_feat = self.featnet(x)
        depth_feat = x
        cam_params = torch.cat([intrins.reshape(B*N,-1),
                               post_rots.reshape(B*N,-1),
                               post_trans.reshape(B*N,-1),
                               rots.reshape(B*N,-1),
                               trans.reshape(B*N,-1)],dim=1)
        depth_feat = self.se(depth_feat, cam_params)
        depth_feat = self.extra_depthnet(depth_feat)[0]
        depth_feat = self.dcn(depth_feat)
        depth_digit = self.depthnet(depth_feat)
        depth_prob = depth_digit.softmax(dim=1)
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        B, N, _ = trans.shape

        intrins_transform = torch.eye(4, 4).to(trans.device)
        intrins_transform = intrins_transform[None, None, :,:].repeat(B, N, 1, 1)
        post_transform = intrins_transform.clone()
        intrins_transform = intrins_transform.clone()
        transform = intrins_transform.clone()

        post_transform[:,:,:3,:3] = post_rots
        # post_transform[:,:,3,:3] = post_trans
        post_transform[:,:,:3,3] = post_trans

        transform[:,:,:3,:3] = rots
        # transform[:,:,3,:3] = trans
        transform[:,:,:3,3] = trans

        intrins_transform[:,:,:3,:3] = intrins
        pseudo_points = self.points.clone()[None, None,:,:].expand(B, N, -1, -1)
        pseudo_points = torch.cat([
            pseudo_points,
            torch.ones_like(pseudo_points)[..., 0:1]], dim=-1)
        lidar2img_transform = intrins_transform.matmul(torch.inverse(transform))
        pseudo_points_2d = lidar2img_transform.view(B, N, 1, 4, 4).matmul(pseudo_points.unsqueeze(-1))
        pseudo_points_2d = pseudo_points_2d.squeeze(-1)

        pseudo_points_2d[...,:2] = pseudo_points_2d[...,:2] / (pseudo_points_2d[...,2:3].abs() + 1e-2)

        img2augimg_transform = post_transform.view(B, N, 1, 4, 4)
        pseudo_points_2d = img2augimg_transform.matmul(pseudo_points_2d.unsqueeze(-1))
        pseudo_points_2d = pseudo_points_2d.squeeze(-1)
        # pseudo_points_2d do the normalization; -> check the dim;
        # also normalization with depth;
        pseudo_points_2d_grid = pseudo_points_2d.clone()
        pseudo_points_2d_grid[..., 0] /= self.ogfW
        pseudo_points_2d_grid[..., 1] /= self.ogfH
        pseudo_points_2d_grid[... ,2] -= self.grid_config['dbound'][0]
        pseudo_points_2d_grid[..., 2] /= self.D
        
        pseudo_points_2d_grid = pseudo_points_2d_grid*2 - 1
        pseudo_points_2d_grid = pseudo_points_2d_grid[...,:3]

        volume = F.grid_sample(
                volume.reshape(B*N, self.numC_Trans, self.D, H, W), 
                pseudo_points_2d_grid.reshape(B*N, 1, 1, -1, 3),
                mode='bilinear',
                align_corners=True)

        valid = (pseudo_points_2d[..., 0] > 0) & (pseudo_points_2d[..., 0] < self.ogfW)
        valid = valid & (pseudo_points_2d[..., 1] > 0) & (pseudo_points_2d[..., 1] < self.ogfH)
        valid = valid & (pseudo_points_2d[..., 2] > self.grid_config['dbound'][0]) & (pseudo_points_2d[..., 2] < self.grid_config['dbound'][1])

        valid = valid.float().detach().reshape(B*N, 1, 1, 1, -1) 
        volume = volume * valid
        bev_feat = volume.reshape(B, N, -1, self.n_voxels[2], self.n_voxels[1], self.n_voxels[0])
        valid = valid.reshape(B, N, -1, self.n_voxels[2], self.n_voxels[1], self.n_voxels[0])
        valid = valid.expand(-1, -1, self.numC_Trans, -1, -1, -1)
        bev_feat = bev_feat.sum(1) / valid.sum(1).clamp(min=1e-2)

        if self.bev_pool:
            bev_feat = bev_feat.mean(dim=2)
        # pseudo_points *
        # do the valid sample correction
        return bev_feat, depth_digit