import torch
from torch import nn
import torch.nn.functional as F
from mmdet.models import NECKS
# from det3d.ops.build_cost_volume import build_cost_volume
import mmcv
def convbn(in_planes,
           out_planes,
           kernel_size,
           stride,
           pad,
           dilation=1,
           gn=False,
           groups=32):
    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=dilation if dilation > 1 else pad,
                  dilation=dilation,
                  bias=False),
        mmcv.SyncBatchNorm(out_planes) if not gn else nn.GroupNorm(
            groups, out_planes))


def convbn_3d(in_planes,
              out_planes,
              kernel_size,
              stride,
              pad,
              gn=False,
              groups=32):
    return nn.Sequential(
        nn.Conv3d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  padding=pad,
                  stride=stride,
                  bias=False),
        nn.BatchNorm3d(out_planes) if not gn else nn.GroupNorm(
            groups, out_planes))

class upconv_module(nn.Module):
    def __init__(self, in_channels, up_channels):
        super(upconv_module, self).__init__()
        self.num_stage = len(in_channels) - 1
        self.conv = nn.ModuleList()
        self.redir = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            self.conv.append(
                convbn(in_channels[0] if stage_idx == 0 else up_channels[stage_idx - 1], up_channels[stage_idx], 3, 1, 1, 1)
            )
            self.redir.append(
                convbn(in_channels[stage_idx + 1], up_channels[stage_idx], 3, 1, 1, 1)
            )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, feats):
        x = feats[0]
        for stage_idx in range(self.num_stage):
            x = self.conv[stage_idx](x)
            redir = self.redir[stage_idx](feats[stage_idx + 1])
            x = F.relu(self.up(x) + redir)
        return x


class hourglass(nn.Module):
    def __init__(self, inplanes, gn=False):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(inplanes,
                      inplanes * 2,
                      kernel_size=3,
                      stride=2,
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               stride=1,
                               pad=1,
                               gn=gn)

        self.conv3 = nn.Sequential(
            convbn_3d(inplanes * 2,
                      inplanes * 2,
                      kernel_size=3,
                      stride=2,
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn_3d(inplanes * 2,
                      inplanes * 2,
                      kernel_size=3,
                      stride=1,
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes *
                           2) if not gn else nn.GroupNorm(32, inplanes *
                                                          2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2,
                               inplanes,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes)
            if not gn else nn.GroupNorm(32, inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu,
                          inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class hourglass2d(nn.Module):
    def __init__(self, inplanes, gn=False):
        super(hourglass2d, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes,
                   inplanes * 2,
                   kernel_size=3,
                   stride=2,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv2 = convbn(inplanes * 2,
                            inplanes * 2,
                            kernel_size=3,
                            stride=1,
                            pad=1,
                            dilation=1,
                            gn=gn)

        self.conv3 = nn.Sequential(
            convbn(inplanes * 2,
                   inplanes * 2,
                   kernel_size=3,
                   stride=2,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn(inplanes * 2,
                   inplanes * 2,
                   kernel_size=3,
                   stride=1,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes *
                           2) if not gn else nn.GroupNorm(32, inplanes *
                                                          2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2,
                               inplanes,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes)
            if not gn else nn.GroupNorm(32, inplanes))  # +x

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu,
                          inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post

@NECKS.register_module()
class LigaStereoNeck(nn.Module):
    # the neck that extracts feature for building stereo cost volume.
    '''
        concat the features from different resnet layers.

    '''

    def __init__(self,
                    in_dims=[3, 64, 128, 128, 128],
                    with_upconv=True,
                    start_level=2,
                    cat_img_feature=True,
                    sem_dim=[128, 32],
                    stereo_dim=[32, 32],
                    spp_dim=32,
                    use_GN=True,):
        super().__init__()

        self.in_dims = in_dims
        self.with_upconv = with_upconv
        self.start_level = start_level
        self.cat_img_feature = cat_img_feature

        self.sem_dim = sem_dim
        self.stereo_dim = stereo_dim

        self.spp_dim = spp_dim
        self.use_GN = use_GN

        self.spp_branches = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(s, stride=s),
                convbn(self.in_dims[-1],
                    self.spp_dim,
                    1, 1, 0,
                    gn=self.use_GN,
                    groups=min(32, self.spp_dim)),
                nn.ReLU(inplace=True))
            for s in [(64, 64), (32, 32), (16, 16), (8, 8)]])
            # for s in [(32, 32), (16, 16), (8, 8)]])
        concat_dim = self.spp_dim * len(self.spp_branches) + sum(self.in_dims[self.start_level:])

        if self.with_upconv:
            assert self.start_level == 2
            self.upconv_module = upconv_module([concat_dim, self.in_dims[1], self.in_dims[0]], [64, 32])
            stereo_dim = 32
        else:
            stereo_dim = concat_dim
            assert self.start_level >= 1
        self.lastconv = nn.Sequential(
            convbn(stereo_dim, self.stereo_dim[0], 3, 1, 1, gn=self.use_GN),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stereo_dim[0], self.stereo_dim[1],
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      bias=False))

        if self.cat_img_feature:
            self.rpnconv = nn.Sequential(
                convbn(concat_dim, self.sem_dim[0], 3, 1, 1, 1, gn=self.use_GN),
                nn.ReLU(inplace=True),
                convbn(self.sem_dim[0], self.sem_dim[1], 3, 1, 1, gn=self.use_GN),
                nn.ReLU(inplace=True)
            )


    def forward(self, feats):
        feat_shape = tuple(feats[self.start_level].shape[2:])
        assert len(feats) == len(self.in_dims)
        spp_branches = []
        for branch_module in self.spp_branches:
            x = branch_module(feats[-1])
            x = F.interpolate(
                x, feat_shape,
                mode='bilinear',
                align_corners=True)
            spp_branches.append(x)

        concat_feature = torch.cat((*feats[self.start_level:], *spp_branches), 1)
        stereo_feature = concat_feature

        if self.with_upconv:
            stereo_feature = self.upconv_module([stereo_feature, feats[1], feats[0]])

        stereo_feature = self.lastconv(stereo_feature)

        if self.cat_img_feature:
            sem_feature = self.rpnconv(concat_feature)
        else:
            sem_feature = None

        return stereo_feature, sem_feature



@NECKS.register_module()
class LigaCostVolumeNeck(nn.Module):
    '''
    Neck for the cost volume
    input_dim: int
    cv_dim: dim of latent features in the cost volume
    use_GN: use GN or BN
    num_hg: number of the hour glass

    '''
    def __init__(self,input_dim,
                      cv_dim,
                      use_GN=True,
                      num_hg=1):
        super().__init__()

        self.dres0 = nn.Sequential(
            convbn_3d(input_dim, cv_dim, 3, 1, 1, gn=use_GN),
            nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(
            convbn_3d(cv_dim, cv_dim, 3, 1, 1, gn=use_GN))

        self.hg_stereo = nn.ModuleList()
        assert num_hg > 0, 'at least one hourglass'
        for _ in range(num_hg):
            self.hg_stereo.append(hourglass(cv_dim, gn=use_GN))

    def forward(self, cost_raw):
        cost0 = self.dres0(cost_raw)
        cost0 = self.dres1(cost0) + cost0
        if len(self.hg_stereo) > 0:
            all_costs = []
            cur_cost = cost0
            for hg_stereo_module in self.hg_stereo:
                cost_residual, _, _ = hg_stereo_module(cur_cost, None, None)
                cur_cost = cur_cost + cost_residual
                all_costs.append(cur_cost)
        else:
            all_costs = [cost0]

        return all_costs



@NECKS.register_module()
class LigaVoxelNeck(nn.Module):
    '''
    Neck for 3D voxel in liga stereo
    '''
    def __init__(self, num_3dconvs,
                        input_dim,
                        rpn3d_dim,
                        use_GN=True):
        super().__init__()
        self.rpn3d_convs = []
        for i in range(num_3dconvs):
            self.rpn3d_convs.append(
                nn.Sequential(
                    convbn_3d(input_dim if i==0 else rpn3d_dim,
                            rpn3d_dim, 3, 1, 1, gn=use_GN),
                nn.ReLU(inplace=True)))
        self.rpn3d_convs = nn.Sequential(*self.rpn3d_convs)
        self.rpn3d_pool = torch.nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))
        self.num_3d_features = rpn3d_dim

    def forward(self, voxel):
        voxel_nopool = self.rpn3d_convs(voxel)
        voxel = self.rpn3d_pool(voxel_nopool)
        return voxel_nopool, voxel


@NECKS.register_module()
class HeightCompression(nn.Module):
    def __init__(self, num_bev_features, sparse_input=False):
        super().__init__()
        self.num_bev_features = num_bev_features
        self.sparse_input = sparse_input

    def forward(self, output):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        if self.sparse_input:
            raise NotImplementatedError
            # encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            # spatial_features = encoded_spconv_tensor.dense()
            # batch_dict['volume_features'] = spatial_features
        else:
            spatial_features = output['voxel_features']
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        output['spatial_features'] = spatial_features
        # if self.sparse_input:
        #     batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        # else:
        output['spatial_features_stride'] = 1
        return output


@NECKS.register_module()
class HourglassBEVNeck(nn.Module):
    def __init__(self, input_channels,
                    num_channels,
                    flip_coordinate=False,
                    use_GN=True):
        super().__init__()
        self.num_channels = num_channels

        self.flip_coordinate = flip_coordinate
        self.rpn3d_conv2 = nn.Sequential(
            convbn(input_channels, self.num_channels, 3, 1, 1, 1, gn=use_GN),
            nn.ReLU(inplace=True))
        self.rpn3d_conv3 = hourglass2d(self.num_channels, gn=use_GN)
        self.num_bev_features = self.num_channels

    def forward(self, output):
        spatial_features = output['spatial_features']
        x = self.rpn3d_conv2(spatial_features)

        output['spatial_features_2d_prehg'] = x
        x = self.rpn3d_conv3(x, None, None)[0]
        output['spatial_features_2d'] = x
        return output



@NECKS.register_module()
class BuildCostVolume(nn.Module):
    def __init__(self, volume_types=[]):
        super().__init__()
        self.volume_types = volume_types

    def get_dim(self, feature_channel):
        d = 0
        volumes = []
        for volume_type in volume_types:
            if volume_type["type"] == "concat":
                d += feature_channel * 2
            else:
                raise NotImplementedError
        return d

    def forward(self, left, right, left_raw, right_raw, shift):
        volumes = []
        for volume_type in self.volume_types:
            if volume_type["type"] == "concat":
                downsample = volume_type["downsample"]
                volumes.append(build_cost_volume(left, right, shift, downsample))
            else:
                raise NotImplementedError
        if len(volumes) > 1:
            return torch.cat(volumes, dim=1)
        else:
            return volumes[0]
    def __repr__(self):
        tmpstr = self.__class__.__name__
        return tmpstr
