import torch
from torch import nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models import NECKS
import math


class Basic3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm(x)
        
        if self.out_channels != self.in_channels:
            residual = self.identity(residual)
        x = x + residual
        x = self.relu(x)

        return x 


class Post2D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Post2D_block, self).__init__()
        self.block1 = Basic3x3(in_channels, in_channels * 2)
        self.downsample1 = nn.MaxPool2d(2, 2)
        self.block2 = Basic3x3(in_channels * 2, in_channels * 2)
        self.downsample2 = nn.MaxPool2d(2, 2)

        self.fuse = Basic3x3(in_channels * 5, out_channels)

    def forward(self, x):
        '''args: n c h w
        '''
        n, c, _H, _W = x.shape
        x1 = self.block1(x) 
        x1 = self.downsample1(x1) # n 2c h/2 w/2
        x2 = self.block2(x1)
        x2 = self.downsample2(x2) # n 2c h/4 w/4

        x1 = torch.nn.functional.interpolate(x1, (_H, _W), mode='bilinear', align_corners=True)
        x2 = torch.nn.functional.interpolate(x2, (_H, _W), mode='bilinear', align_corners=True)
        x = torch.cat([x, x1, x2], dim=1) # c + 2c + 2c
        x = self.fuse(x)

        return x

class Conv3x3x3_Residual(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm='BN', drop=0):
        super(Conv3x3x3_Residual, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
  
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, 3, 1, 1),
            nn.BatchNorm3d(planes),
        )
        self.relu = nn.ReLU(inplace=True)

        self.mchannel = nn.Sequential(
            nn.Conv3d(inplanes, planes, 1, 1),
            nn.BatchNorm3d(planes)
        )

    def forward(self, x):
        identity = x
        

        out = self.conv1(x)

        if self.planes != self.inplanes:
            identity = self.mchannel(identity)

        out += identity
        out = self.relu(out)

        return out

    def init_weights(self):
        pass


def _get_conv(in_channels, out_channels, stride=(1, 1, 2), padding=(1, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


@NECKS.register_module()
class InteracitveNeckLSS(nn.Module):
    def __init__(self, in_channels, out_channels, n_voxels, voxel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size

        self.block = nn.ModuleList([
            Conv3x3x3_Residual(in_channels, in_channels),
            Conv3x3x3_Residual(in_channels * 2, in_channels * 2),
            Conv3x3x3_Residual(in_channels * 4, out_channels),
        ])
        self.attn_block = nn.ModuleList([
            nn.Conv3d(in_channels, in_channels * 9, 3, 1, 1, groups=in_channels),
            nn.Conv3d(in_channels * 2, in_channels * 2 * 9, 3, 1, 1, groups=in_channels * 2),
            nn.Conv3d(out_channels, out_channels * 9, 3, 1, 1, groups=out_channels)
        ])

        self.norm1 = nn.ModuleList([
            nn.BatchNorm3d(in_channels),
            nn.BatchNorm3d(in_channels * 2),
            nn.BatchNorm3d(out_channels)
        ])

        self.relu = nn.functional.relu
        
        self.downsample1 = nn.ModuleList([
            _get_conv(in_channels, in_channels * 2),
            _get_conv(in_channels * 2, in_channels * 4),
            _get_conv(out_channels, out_channels, 1, (1, 1, 0))
        ])
        self.downsample2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 2, 3, 2, 1),
                nn.BatchNorm2d(in_channels * 2),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels * 4, 3, 2, 1),
                nn.BatchNorm2d(in_channels * 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
        ])
        self.n_height = [12, 6, 3]
        self.size_height = [0.32, 0.64, 1.28]

        self.conv1x1 = nn.Conv2d(128, 256, 1, 1)

        self.depth_prob = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, 20, 1)
        )

        self.post_bev = Post2D_block(128, 256)
    
    def forward(self, input, img_metas, img_shape):
        '''x: [tensor(n, 6, c, h, w)]
            img_shape: img shape
        '''
        x = input[0] #FOV feature [n 6 c h w]
        stride = img_shape[-1] / input[0].shape[-1]
        height = int(img_metas[0]['img_shape'][0] // stride)
        width = int(img_metas[0]['img_shape'][1] // stride)
        
        x = x[0, :, :, :height, :width] # 6, c, h, w

        x_dp = self.depth_prob(x) # 6, c, h, w
        x_dp = torch.nn.functional.softmax(x_dp, 1) # 6, D, h, w
    
        outer_x = x_dp.unsqueeze(1) * x.unsqueeze(2) # 6, C, D, h, w


        origin = torch.tensor(img_metas[0]['lidar2img']['origin'])
        voxel, _ = self.FOV2VOXEL_3d(outer_x, stride, img_metas, self.n_voxels, self.voxel_size, origin) #n, c, x, y, z
        # print('voxel 0: ', voxel.sum())

        for i in range(1):
            voxel = self.block[i](voxel) 
            attn = self.attn_block[i](voxel) # N, C * 9, X, Y, Z 

            # import pdb 
            # pdb.set_trace() # 5168MB

            # get unfold of fov
            x_unfold = torch.nn.functional.unfold(x, kernel_size=3, stride=1, dilation=1, padding=1)
            
            # 5464MB

            batch_size, C, _H, _W = x.shape
            batch_size, C_k_k, L = x_unfold.shape
            x_unfold = x_unfold.view(batch_size, C_k_k, _H, _W) # batchsize, C * 9, H, W

            # get unfold voxel
            if i > 0:
                stride = stride * 2

            # import pdb
            # pdb.set_trace() # 5464MB
            
            voxel_unfold, valids = self.FOV2VOXEL(x_unfold, stride, img_metas, 
                    [self.n_voxels[0], self.n_voxels[1], self.n_height[i]],
                    [self.voxel_size[0], self.voxel_size[1], self.size_height[i]], 
                    origin)  # N, C * 9, X, Y, Z

            # 12810MB

            batchsize, CC, _X, _Y, _Z = voxel_unfold.shape
            batchsize = 1

            # dynamic conv
            voxel_unfold = voxel_unfold.view(batchsize, CC//9, 9, _X, _Y, _Z) * attn.view(batchsize, CC//9, 9, _X, _Y, _Z).sigmoid()
            voxel_unfold = voxel_unfold.sum(2) # batch_size, C, _X, _Y, _Z

            # import pdb
            # pdb.set_trace() # 13810MB

            voxel = self.norm1[i](voxel_unfold) + voxel
            voxel = self.relu(voxel)

            voxel = self.downsample1[i](voxel)
            x = self.downsample2[i](x)
        # import pdb
        # pdb.set_trace()

        voxel = voxel.mean(4)
        # voxel = self.conv1x1(voxel).unsqueeze(4)
        voxel = self.post_bev(voxel).unsqueeze(4)
        return [voxel.squeeze(4).transpose(-1, -2)], valids.squeeze(4).transpose(-1, -2)

    def init_weights(self):
        pass
    
    def FOV2VOXEL(self, feature, stride, img_metas, n_voxels, voxel_size, origin):
        all_volume = []
        all_valid = []
        # for idx, (x, img_meta) in enumerate(zip(feature, img_metas)): #TODO support batch size > 1
        x = feature
        img_meta = img_metas[0]
        projection = self._compute_projection(img_meta, stride).to(x.device)
        points = self.get_points(n_voxels, voxel_size, origin).to(x.device)
        volume, valid = self.backproject(x, points, projection)
        all_volume.append(volume)
        all_valid.append(valid)
      
        all_volume = torch.stack(all_volume).sum(dim=1)
        all_valid = torch.stack(all_valid).sum(dim=1)

        mask = (all_valid != 0).repeat(1, all_volume.shape[1], 1, 1, 1)
        all_volume = all_volume * mask

        volume = all_volume / (all_valid + 1e-10)

        return volume, all_valid

    def FOV2VOXEL_3d(self, feature, stride, img_metas, n_voxels, voxel_size, origin):
        all_volume = []
        all_valid = []
        # for idx, (x, img_meta) in enumerate(zip(feature, img_metas)): #TODO support batch size > 1
        x = feature
        img_meta = img_metas[0]
        projection = self._compute_projection(img_meta, stride).to(x.device)
        points = self.get_points(n_voxels, voxel_size, origin).to(x.device)
        volume, valid = self.backproject_3d(x, points, projection)
        all_volume.append(volume)
        all_valid.append(valid)
      
        all_volume = torch.stack(all_volume).sum(dim=1)
        all_valid = torch.stack(all_valid).sum(dim=1)

        mask = (all_valid != 0).repeat(1, all_volume.shape[1], 1, 1, 1)
        all_volume = all_volume * mask

        volume = all_volume / (all_valid + 1e-10)

        return volume, all_valid

    @staticmethod    
    def backproject(features, points, projection):
        n_images, n_channels, height, width = features.shape
        n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
        points = points.view(1, 3, -1).expand(n_images, 3, -1)
        points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
        points_2d_3 = torch.bmm(projection, points)
        x = points_2d_3[:, 0] / (points_2d_3[:, 2] + 1e-10) # 1. add round
        y = points_2d_3[:, 1] / (points_2d_3[:, 2] + 1e-10)
        z = points_2d_3[:, 2]
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0) # 2. add valid
        volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)
        x = (x.float() / features.shape[-1]) * 2 - 1
        y = (y.float() / features.shape[-2]) * 2 - 1
        grid = torch.stack([x, y], dim=-1)
        grid = grid.expand(n_images, -1, -1)
        volume = torch.nn.functional.grid_sample(features, grid.unsqueeze(1), align_corners=True, padding_mode="zeros").squeeze(2)
        #
        # for i in range(n_images):
            # volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
        volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
        valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
        return volume, valid    
    
    @staticmethod
    def backproject_3d(features, points, projection):
        n_images, n_channels, depth, height, width = features.shape
        n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
        points = points.view(1, 3, -1).expand(n_images, 3, -1)
        points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
        points_2d_3 = torch.bmm(projection, points)
        x = points_2d_3[:, 0] / (points_2d_3[:, 2] + 1e-10) # 1. add round
        y = points_2d_3[:, 1] / (points_2d_3[:, 2] + 1e-10)
        z = points_2d_3[:, 2]
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0) # 2. add valid
        volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)
        x = (x.float() / features.shape[-1]) * 2 - 1
        y = (y.float() / features.shape[-2]) * 2 - 1
        
        z = z - z.min()
        z = z / z.max() * 2 - 1

        grid = torch.stack([x, y, z], dim=-1)
        grid = grid.expand(n_images, -1, -1)
    
        volume = torch.nn.functional.grid_sample(features, grid.unsqueeze(1).unsqueeze(1), align_corners=True, padding_mode="zeros").squeeze(2)
        volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
        valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
        return volume, valid    
    
    @staticmethod
    def get_points(n_voxels, voxel_size, origin):
        points = torch.stack(torch.meshgrid([
            torch.arange(n_voxels[0]), 
            torch.arange(n_voxels[1]), 
            torch.arange(n_voxels[2]) 
        ]))
        n_voxels = torch.tensor(n_voxels)
        voxel_size = torch.tensor(voxel_size)
        origin = torch.tensor(origin)

        new_origin = origin - n_voxels / 2. * voxel_size
        points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
        return points



    @staticmethod
    def _compute_projection(img_meta, stride, angles=None):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio
        # use predicted pitch and roll for SUNRGBDTotal test
        if angles is not None:
            extrinsics = []
            for angle in angles:
                extrinsics.append(get_extrinsics(angle).to(intrinsic.device))
        else:
            extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
        for extrinsic in extrinsics:
            projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)


