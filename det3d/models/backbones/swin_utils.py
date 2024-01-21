# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner.base_module import BaseModule
from torch.nn.modules.utils import _pair as to_2tuple
from collections import OrderedDict


# Modified from Pytorch-Image-Models
class PatchEmbed(BaseModule):
    """Image to Patch Embedding V2.

    We use a conv layer to implement PatchEmbed.
    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (dict, optional): The config dict for conv layers type
            selection. Default: None.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Default to be equal with kernel_size).
        padding (int): The padding length of embedding conv. Default: 0.
        dilation (int): The dilation rate of embedding conv. Default: 1.
        pad_to_patch_size (bool, optional): Whether to pad feature map shape
            to multiple patch size. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type=None,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 dilation=1,
                 pad_to_patch_size=True,
                 norm_cfg=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__()

        self.embed_dims = embed_dims
        self.init_cfg = init_cfg

        if stride is None:
            stride = kernel_size

        self.pad_to_patch_size = pad_to_patch_size

        # The default setting of patch size is equal to kernel size.
        patch_size = kernel_size
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        elif isinstance(patch_size, tuple):
            if len(patch_size) == 1:
                patch_size = to_2tuple(patch_size[0])
            assert len(patch_size) == 2, \
                f'The size of patch should have length 1 or 2, ' \
                f'but got {len(patch_size)}'

        self.patch_size = patch_size

        # Use conv layer to embed
        conv_type = conv_type or 'Conv2d'
        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # TODO: Process overlapping op
        if self.pad_to_patch_size:
            # Modify H, W to multiple of patch size.
            if H % self.patch_size[0] != 0:
                x = F.pad(
                    x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            if W % self.patch_size[1] != 0:
                x = F.pad(
                    x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))

        x = self.projection(x)
        self.DH, self.DW = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x





def swin_convert(ckpt):
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt[new_k] = new_v

    return new_ckpt


def vit_convert(ckpt):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        if k.startswith('norm'):
            new_k = k.replace('norm.', 'ln1.')
        elif k.startswith('patch_embed'):
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        elif k.startswith('blocks'):
            if 'norm' in k:
                new_k = k.replace('norm', 'ln')
            elif 'mlp.fc1' in k:
                new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = k.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn.qkv' in k:
                new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
            elif 'attn.proj' in k:
                new_k = k.replace('attn.proj', 'attn.attn.out_proj')
            else:
                new_k = k
            new_k = new_k.replace('blocks.', 'layers.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt
