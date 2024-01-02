import torch
from torch import nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models import NECKS

import math
def get_conv2d(in_channels, out_channels, stride=(1, 1), padding=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )



@NECKS.register_module()
class ImVoxelNeck(nn.Module):
    def __init__(self,
                 channels,
                 out_channels,
                 down_layers,
                 up_layers,
                 conditional,
                 inverse_map=True):
        super().__init__()
        self.model = EncoderDecoder(channels=channels,
                                    layers_down=down_layers,
                                    layers_up=up_layers,
                                    cond_proj=conditional)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in channels])
        self.inverse_map = inverse_map


    @auto_fp16()
    def forward(self, x):
        x = self.model.forward(x)[::-1]
        if self.inverse_map:
            x = x[::-1]
        return [self.conv_blocks[i](x[i]) for i in range(len(x))]

    def init_weights(self):
        pass


@NECKS.register_module()
class KittiImVoxelNeck(nn.Module):
    def __init__(self, in_channels, out_channels, last_downsample=True):
        super().__init__()
        if last_downsample:
            last_downsample = (1, 1, 1)
            last_pad = (1, 1, 0)
        else:
            last_downsample = (1, 1, 2)
            last_pad = (1, 1, 1)

        self.model = nn.Sequential(
            BasicBlock3d(in_channels, in_channels),
            self._get_conv(in_channels, in_channels * 2),
            BasicBlock3d(in_channels * 2, in_channels * 2),
            self._get_conv(in_channels * 2, in_channels * 4),
            BasicBlock3d(in_channels * 4, in_channels * 4),
            # todo: padding should be (1, 1, 0) here
            self._get_conv(in_channels * 4, out_channels, last_downsample, last_pad)
        )

    @staticmethod
    def _get_conv(in_channels, out_channels, stride=(1, 1, 2), padding=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    @auto_fp16()
    def forward(self, input):
        output = []
        for x in input:
            x = self.model.forward(x)
            assert x.shape[-1] == 1
            output.append(x[..., 0].transpose(-1, -2)) # transpose in here.
        return output


    def init_weights(self):
        pass



@NECKS.register_module()
class KittiPSPImVoxelNeck(nn.Module):
    def __init__(self, in_channels, out_channels, last_downsample=True):
        super().__init__()
        if last_downsample:
            last_downsample = (1, 1, 1)
            last_pad = (1, 1, 0)
        else:
            last_downsample = (1, 1, 2)
            last_pad = (1, 1, 1)
        self.pre_model = BasicBlock3d(in_channels, in_channels)

        self.psp_model = PSPBlock3d(in_channels, in_channels)
        self.post_model = nn.Sequential(
            self._get_conv(in_channels, in_channels * 2),
            BasicBlock3d(in_channels * 2, in_channels * 2),
            self._get_conv(in_channels * 2, in_channels * 4),
            BasicBlock3d(in_channels * 4, in_channels * 4),
            # todo: padding should be (1, 1, 0) here
            self._get_conv(in_channels * 4, out_channels, last_downsample, last_pad)
        )

    @staticmethod
    def _get_conv(in_channels, out_channels, stride=(1, 1, 2), padding=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    @auto_fp16()
    def forward(self, input):
        output = []
        for x in input:
            x = self.pre_model(x)
            x = self.psp_model(x)
            x = self.post_model(x)
            assert x.shape[-1] == 1
            output.append(x[..., 0].transpose(-1, -2)) # transpose in here.
        return output


    def init_weights(self):
        pass





@NECKS.register_module()
class IdentityNeck(nn.Module):
    def __init__(self):
        super().__init__()
    @auto_fp16()
    def forward(self, x):
        return x

    def init_weights(self):
        pass



@NECKS.register_module()
class InverseNeck(nn.Module):
    def __init__(self):
        super().__init__()
    @auto_fp16()
    def forward(self, x):
        return x[::-1]

    def init_weights(self):
        pass



@NECKS.register_module()
class NeckConv(nn.Module):
    def __init__(self, in_channels, out_channels, light_weight=False):
        super().__init__()
        self.layer1 = nn.Sequential(
            BasicBlock2d(in_channels, in_channels),
            self._get_conv2d(in_channels, in_channels * 2))

        self.layer2 = nn.Sequential(
            BasicBlock2d(in_channels*2, in_channels * 2),
            self._get_conv2d(in_channels*2, in_channels*4))
        if light_weight is False:
            self.layer3 = nn.Sequential(
                BasicBlock2d(in_channels*4, in_channels*4),
                self._get_conv2d(in_channels*4, out_channels))
        else:
            self.layer3 = nn.Identity()
    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        output = []
        output.append(x)
        x = self.layer1(x)
        output.append(x)
        x = self.layer2(x)
        output.append(x)
        x = self.layer3(x)
        output.append(x)
        return output

    @staticmethod
    def _get_conv2d(in_channels, out_channels, stride=(1, 1), padding=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


@NECKS.register_module()
class Trans2d3dNeck(nn.Module):
    def __init__(self, in_channels,
                     out_channels,
                     num_layers = 4,
                     bev_shape=[],
                     embedding_type = "semantic",
                     norm = None,
                     dropout=0.,
                     n_head=4,
                     key_receptive_size=1,
                     position_encoding=False,
                     device="cuda"):
        super().__init__()
        assert in_channels == out_channels # currently only support this mode
        self.module = []
        for layer in range(num_layers):
            if layer == 0:
                self.module.append(
                    BasicBevFovTrans(in_channels,
                                     out_channels,
                                     bev_shape,
                                     embedding_type,
                                     norm=norm,
                                     dropout=dropout,
                                     n_head=n_head,
                                     key_receptive_size=key_receptive_size,
                                     position_encoding=position_encoding,
                                     need_embedding=True))
            else:
                self.module.append(
                    BasicBevFovTrans(in_channels,
                                     out_channels,
                                     bev_shape,
                                     embedding_type,
                                     norm=norm,
                                     dropout=dropout,
                                     n_head=n_head,
                                     key_receptive_size=key_receptive_size,
                                     position_encoding=position_encoding,
                                     need_embedding=False))

        # if embedding_type == "semantic":
        #     bev_size = bev_shape[0] * bev_shape[1]
        #     self.query_embedding = nn.Embedding(bev_size, in_channels)
        #     self.base_query = torch.arange(bev_size)
        #     self.base_query = self.base_query.to(device)
        # else:
        #     raise NotImplementedError

        self.module = nn.Sequential(*self.module)

        # self.embedding = nn
    def forward(self, x_3d):
        x_3d = x_3d[0]

        x_bev, x_3d = self.module(x_3d)
        # x_bev.unsqueeze(-1)
        # x_bev = [x_bev[..., 0].transpose(-1, -2)]
        x_bev = [x_bev.permute(0,1,3,2)]
        return x_bev

    def init_weights(self):
        pass




@NECKS.register_module()
class Trans2d3dNeckV2(nn.Module):
    def __init__(self, in_channels,
                     bev_shape=[],
                     embedding_type = "semantic",
                     norm = None,
                     dropout=0.,
                     n_head=4,
                     position_encoding=False,
                     act_type="relu",
                     twice_conv=False,
                     out_conv=False,
                     light_weight=False,
                     key_receptive_size=1,
                     device="cuda"):
        super().__init__()
        # assert in_channels == out_channels # currently only support this mode
        self.module = []
        if twice_conv == "True":
            twice_conv = True
        self.light_weight=light_weight
        self.layer1 = BasicBevFovTrans(in_channels,
                        in_channels,
                        bev_shape,
                        embedding_type,
                        norm=norm,
                        dropout=dropout,
                        n_head=n_head,
                        position_encoding=position_encoding,
                        act_type=act_type,
                        twice_conv=twice_conv,
                        out_conv=out_conv,
                        key_receptive_size=key_receptive_size,
                        need_embedding=True)
        self.layer2 = BasicBevFovTrans(in_channels,
                        in_channels*2,
                        bev_shape,
                        embedding_type,
                        norm=norm,
                        dropout=dropout,
                        n_head=n_head,
                        act_type=act_type,
                        twice_conv=twice_conv,
                        out_conv=out_conv,
                        key_receptive_size=key_receptive_size,
                        position_encoding=position_encoding)
        if self.light_weight is False:
            self.layer3 = BasicBevFovTrans(in_channels*2,
                            in_channels*4,
                            bev_shape,
                            embedding_type,
                            norm=norm,
                            dropout=dropout,
                            n_head=n_head,
                            position_encoding=position_encoding,
                            act_type=act_type,
                            twice_conv=twice_conv,
                            out_conv=out_conv,
                            key_receptive_size=key_receptive_size,
                            need_embedding=False)
            self.layer4 = BasicBevFovTrans(in_channels*4,
                            in_channels*4,
                            bev_shape,
                            embedding_type,
                            norm=norm,
                            dropout=dropout,
                            n_head=n_head,
                            position_encoding=position_encoding,
                            act_type=act_type,
                            twice_conv=twice_conv,
                            out_conv=out_conv,
                            key_receptive_size=key_receptive_size,
                            need_embedding=False)

            self.out_layer = get_conv2d(in_channels*4, in_channels)
        else:
            self.layer3 = nn.Identity()
            self.layer4 = nn.Identity()
            self.out_layer = get_conv2d(in_channels*2, in_channels)
        self.all_layer = [self.layer1, self.layer2, self.layer3, self.layer4]
        # self.embedding = nn
    def forward(self, x_3d):
        for idx, (layer, x_3d_idx) in enumerate(zip(self.all_layer, x_3d)):

            if idx == 0:
                x_bev, _ = layer(x_3d_idx)
            else:
                x_bev, _ = layer((x_bev, x_3d_idx))
            # x_bev, _ = self.layermodu1(x_3d_idx)
            # x_bev, _ = self.layer2(x_3d_idx)
            # x_bev, _
        # x_bev, x_3d = self.module(x_3d)
        # x_bev.unsqueeze(-1)
        # x_bev = [x_bev[..., 0].transpose(-1, -2)]
        x_bev = self.out_layer(x_bev)
        x_bev = [x_bev.permute(0,1,3,2)]

        return x_bev

    def init_weights(self):
        pass





@NECKS.register_module()
class Bev2DConvNeck(nn.Module):
    def __init__(self, in_channels, out_channels,
                     device="cuda"):
        super().__init__()
        # assert in_channels == out_channels # currently only support this mode
        self.module = nn.Sequential(
            BasicBlock2d(in_channels, in_channels),
            get_conv2d(in_channels, in_channels * 2),
            BasicBlock2d(in_channels * 2, in_channels * 2),
            get_conv2d(in_channels * 2, in_channels * 4),
            BasicBlock2d(in_channels * 4, in_channels * 4),
            get_conv2d(in_channels * 4, out_channels, 1, (1, 1))
        )

        # self.embedding = nn
    def forward(self, x_3d):
        x_bev = x_3d[0].mean(-1)
        x_bev = self.module(x_bev)
        x_bev = [x_bev.permute(0,1,3,2)]

        return x_bev

    def init_weights(self):
        pass

@NECKS.register_module()
class CustomNuScenes3DNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels = [64, 64]):
        super().__init__()
        model = []
        init_channels = in_channels
        for i in out_channels:
            model.append(
                BasicBlock3d(init_channels, init_channels),)
            model.append(
                self._get_conv(init_channels, i, stride=(2, 1, 1)))
        self.model = nn.Sequential(*model)
    @staticmethod
    def _get_conv(in_channels, out_channels, stride=(1, 1, 2), padding=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model.forward(x)
        x = x.mean(dim=[2])
        return x

@NECKS.register_module()
class NuScenesImVoxelNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample_bev=True,
                 flip_map=True):
        super().__init__()
        self.flip_map = flip_map
        if downsample_bev == "False":
            downsample_bev = False
        if downsample_bev:
            stride0 = 2
        else:
            if self.flip_map is True:
                stride0 = (1, 1, 2)
            else:
                stride0 = (2, 1, 1)
        if self.flip_map is False:
            stride1 = (2, 1, 1)
            padding2 = (0, 1, 1)
        else:
            stride1 = (1, 1, 2)
            padding2 = (1, 1, 0)
        self.model = nn.Sequential(
            BasicBlock3d(in_channels, in_channels),
            self._get_conv(in_channels, in_channels * 2, stride0, 1),
            BasicBlock3d(in_channels * 2, in_channels * 2),
            self._get_conv(in_channels * 2, in_channels * 4, stride1),
            BasicBlock3d(in_channels * 4, in_channels * 4),
            self._get_conv(in_channels * 4, out_channels, 1, padding2)
        )

    @staticmethod
    def _get_conv(in_channels, out_channels, stride=(1, 1, 2), padding=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    @auto_fp16()
    def forward(self, x):
        x = self.model.forward(x)
        if self.flip_map:
            x = [x[..., 0].transpose(-1, -2)]
        else:
            x = x[:,:, 0]
        return x


    def init_weights(self):
        pass


# Everything below is copied from https://github.com/magicleap/Atlas/blob/master/atlas/backbone3d.py
def get_norm_3d(norm, out_channels):
    """ Get a normalization module for 3D tensors
    Args:
        norm: (str or callable)
        out_channels
    Returns:
        nn.Module or None: the normalization layer
    """

    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm3d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)


def get_norm_2d(norm, out_channels):
    """ Get a normalization module for 3D tensors
    Args:
        norm: (str or callable)
        out_channels
    Returns:
        nn.Module or None: the normalization layer
    """

    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)
def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock3d(nn.Module):
    """ 3x3x3 Resnet Basic Block"""
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm='BN', drop=0):
        super(BasicBlock3d, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride, 1, dilation)
        self.bn1 = get_norm_3d(norm, planes)
        # self.drop1 = nn.Dropout(drop, True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1, 1, dilation)
        self.bn2 = get_norm_3d(norm, planes)
        #self.drop2 = nn.Dropout(drop, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.drop1(out) # drop after both??
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.drop2(out) # drop after both??

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PSPBlock3d(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                groups=1, dilation=1,
                norm='BN', drop=0, downsample_times=4):
        super(PSPBlock3d, self).__init__()

        self.psp_module = nn.ModuleList()
        self.downsample_times = downsample_times
        for i in range(downsample_times):
            self.psp_module.append(
                nn.Sequential(
                    conv3x3x3(inplanes, planes, stride, 1, dilation),
                    get_norm_3d(norm, planes),
                    nn.ReLU(inplace=True),))
        self.output_module = conv1x1x1(planes * downsample_times, planes)

    def forward(self, x):
        features = []
        # input = x
        N, C, X, Y, Z = x.shape
        for i in range(self.downsample_times):

            scale = math.pow(2, i)
            if i != 0:
                scaled_features = F.interpolate(x,
                        (int(X/scale), int(Y/scale), int(Z/scale)),
                        mode="trilinear", align_corners=True)
            else:
                scaled_features = x
            scaled_features = self.psp_module[i](scaled_features)
            if i != 0:
                scaled_features = F.interpolate(scaled_features,
                                    [X, Y, Z], mode="trilinear", align_corners=True)

            features.append(scaled_features)
        features = torch.cat(features, dim=1)
        features = self.output_module(features)
        features = features + x
        return features



class BasicBlock2d(nn.Module):
    """ 3x3x3 Resnet Basic Block"""
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm='BN', drop=0):
        super(BasicBlock2d, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, 1, dilation)
        self.bn1 = get_norm_2d(norm, planes)
        # self.drop1 = nn.Dropout(drop, True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, 1, dilation)
        self.bn2 = get_norm_2d(norm, planes)
        #self.drop2 = nn.Dropout(drop, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.drop1(out) # drop after both??
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.drop2(out) # drop after both??

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock3dV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3dV2, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm3d(out_channels)
        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.stride != 1:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ConditionalProjection(nn.Module):
    """ Applies a projected skip connection from the encoder to the decoder
    When condition is False this is a standard projected skip connection
    (conv-bn-relu).
    When condition is True we only skip the non-masked features
    from the encoder. To maintin scale we instead skip the decoder features.
    This was intended to reduce artifacts in unobserved regions,
    but was found to not be helpful.
    """

    def __init__(self, n, norm='BN', condition=True):
        super(ConditionalProjection, self).__init__()
        # return relu(bn(conv(x)) if mask, relu(bn(y)) otherwise
        self.conv = conv1x1x1(n, n)
        self.norm = get_norm_3d(norm, n)
        self.relu = nn.ReLU(True)
        self.condition = condition

    def forward(self, x, y, mask):
        """
        Args:
            x: tensor from encoder
            y: tensor from decoder
            mask
        """

        x = self.conv(x)
        if self.condition:
            x = torch.where(mask, x, y)
        x = self.norm(x)
        x = self.relu(x)
        return x


class EncoderDecoder(nn.Module):
    """ 3D network to refine feature volumes"""

    def __init__(self, channels=[32,64,128], layers_down=[1,2,3],
                 layers_up=[3,3,3], norm='BN', drop=0, zero_init_residual=True,
                 cond_proj=True):
        super(EncoderDecoder, self).__init__()

        self.cond_proj = cond_proj

        self.layers_down = nn.ModuleList()
        self.proj = nn.ModuleList()

        self.layers_down.append(nn.Sequential(*[
            BasicBlock3d(channels[0], channels[0], norm=norm, drop=drop)
            for _ in range(layers_down[0]) ]))
        self.proj.append( ConditionalProjection(channels[0], norm, cond_proj) )
        for i in range(1,len(channels)):
            layer = [nn.Conv3d(channels[i-1], channels[i], 3, 2, 1, bias=(norm=='')),
                     get_norm_3d(norm, channels[i]),
                     nn.Dropout(drop, True),
                     nn.ReLU(inplace=True)]
            layer += [BasicBlock3d(channels[i], channels[i], norm=norm, drop=drop)
                      for _ in range(layers_down[i])]
            self.layers_down.append(nn.Sequential(*layer))
            if i<len(channels)-1:
                self.proj.append( ConditionalProjection(channels[i], norm, cond_proj) )

        self.proj = self.proj[::-1]

        channels = channels[::-1]
        self.layers_up_conv = nn.ModuleList()
        self.layers_up_res = nn.ModuleList()
        for i in range(1,len(channels)):
            self.layers_up_conv.append( conv1x1x1(channels[i-1], channels[i]) )
            self.layers_up_res.append(nn.Sequential( *[
                BasicBlock3d(channels[i], channels[i], norm=norm, drop=drop)
                for _ in range(layers_up[i-1]) ]))

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each
        # residual block behaves like an identity. This improves the
        # model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock3d):
                    nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        if self.cond_proj:
            valid_mask = (x!=0).any(1, keepdim=True).float()


        xs = []
        for layer in self.layers_down:
            x = layer(x)
            xs.append(x)

        xs = xs[::-1]
        out = []
        for i in range(len(self.layers_up_conv)):
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
            x = self.layers_up_conv[i](x)
            if self.cond_proj:
                scale = 1/2**(len(self.layers_up_conv)-i-1)
                mask = F.interpolate(valid_mask, scale_factor=scale)!=0
            else:
                mask = None
            y = self.proj[i](xs[i+1], x, mask)
            x = (x + y)/2
            x = self.layers_up_res[i](x)

            out.append(x)

        return out


class BasicBevFovTrans(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                bev_shape,
                embedding_type=None,
                norm=None,
                dropout=0.,
                n_head=4,
                max_length=12,
                position_encoding=False,
                act_type = "relu",
                key_receptive_size=1,
                twice_conv=False,
                out_conv = False,
                need_embedding=False):
        super().__init__()
        self.need_embedding = need_embedding
        self.embedding_type = embedding_type
        self.key_receptive_size = key_receptive_size
        # if not ned
        self.max_length = max_length # denote as Z
        if need_embedding:
            if embedding_type == "semantic":
                bev_length = bev_shape[0] * bev_shape[1]
                self.query_embedding = nn.Embedding(bev_length, out_channels)
                self.base_query = torch.arange(bev_length)
            elif embedding_type == "average":
                # bev_length = bev_shap
                pass
            elif embedding_type == "weighted_average":
                self.query_weight = nn.Linear(max_length, 1)
                # self.query_act = nn.ReLU(inplace=True)
            else:
                raise NotImplementedError
        else:
            self.conv = nn.Sequential(
                BasicBlock2d(in_channels, in_channels),
                get_conv2d(in_channels, out_channels))

        if out_conv or out_conv == "True":
            self.out_conv = BasicBlock2d(out_channels, out_channels)
        else:
            self.out_conv = nn.Identity()
        self.lift_attn = nn.MultiheadAttention(out_channels, n_head, dropout=dropout)
        if position_encoding:
            self.position_encoding = SinePositionEncoding(max_length, out_channels,)
        else:
            self.position_encoding = None
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "ln":
            self.norm = nn.LayerNorm(bev_shape)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            raise NotImplementedError

        if act_type == "relu":
            self.relu = nn.ReLU(inplace=False)
        elif act_type == "gelu":
            self.relu = nn.GELU()
    def forward(self, x):
        # self
        if self.need_embedding:
            input = None
            x_3d = x
            N, C = x_3d.shape[:2]
            Z = self.max_length
            if self.embedding_type == "semantic":
                self.base_query = self.base_query.to(x.device)
                x_bev = self.query_embedding(self.base_query.unsqueeze(0).expand(N, -1))
            elif self.embedding_type == "average":
                x_bev = x_3d.clone().reshape(N,C, -1, Z).mean(-1, keepdim=True)
                x_bev = x_bev.permute(3, 0,2, 1)
            elif self.embedding_type == "weighted_average":
                x_bev = x_3d.clone().reshape(-1, Z)
                x_bev = self.query_weight(x_bev)
                # x_bev = self.query_act(x_bev)
                x_bev = x_bev.reshape(N, C, -1).permute(0,2,1)
                # x_bev = x_bev.permute
            x_bev = x_bev.reshape(1, -1, C)
        else:
            x_bev, x_3d = x
            x_bev = self.conv(x_bev)
            input = x_bev

            N, C = x_3d.shape[:2]
            x_bev = x_bev.reshape(N, C, -1).permute(0, 2, 1).reshape(1, -1, C)

        N, C, X, Y, Z = x_3d.shape
        if self.key_receptive_size == 1:
            key = x_3d.clone().reshape(N, C, -1, Z)
        elif self.key_receptive_size == 5:
            x_3d_pad = F.pad(x_3d, (0,0,1,1,1,1))
            key = []
            for i, j in zip([0, 1,2, 0,2], [0,1,0,2,2]):
                    key.append(x_3d_pad[:,:,i:i+X,j:j+Y].reshape(N, C, -1, Z))
            key = torch.cat(key, dim=-1)
        elif self.key_receptive_size == 9:
            x_3d = F.pad(x_3d, (0,0,1,1,1,1))
            key = []
            for i in [0,1,2]:
                for j in [0,1,2]:
                    key.append(x_3d[:,:,i:i+X,j:j+Y].reshape(N, C, -1, Z))
            key = torch.cat(key, dim=-1)
        else:
            raise NotImplementedError
        key = key.permute(3, 0, 2, 1).reshape(self.key_receptive_size * Z, -1, C)
        if self.position_encoding is not None:
            key = self.position_encoding(key)
        x_bev = self.lift_attn(x_bev.contiguous(), key.contiguous(), key.contiguous())[0]

        x_bev = x_bev.reshape(N, X, Y, C).permute(0, 3, 1, 2).contiguous()

        if self.norm is not None:
            x_bev = self.norm(x_bev)


        x_bev = self.out_conv(x_bev)

        if input is not None:
            x_bev = x_bev + input
        x_bev = self.relu(x_bev)

        return x_bev, x_3d



class SinePositionEncoding(nn.Module):
    def __init__(self, length, channel_size):
        super().__init__()

        pe = torch.zeros(length, channel_size)

        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, channel_size, 2).float() * -(math.log(10000.0) / channel_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: shape of (max_len, batch, d_model)
        """
        x = x + self.pe[x.size(0)-1,: :]
        return x
