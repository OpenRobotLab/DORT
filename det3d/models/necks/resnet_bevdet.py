from torch import nn
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
import torch.utils.checkpoint as checkpoint

from mmdet.models import NECKS
from mmdet3d.models.builder import build_neck
from mmcv.runner import BaseModule
from mmdet.models import build_backbone

@NECKS.register_module()
class ResNetForBEVDet(nn.Module):
    def __init__(self,
                numC_input,
                num_layer=[2,2,2],
                num_channels=None, stride=[2,2,2],
                backbone_output_ids=None,
                norm_cfg=dict(type='BN'),
                with_cp=False,
                block_type='Basic',
                neck=None,):
        super(ResNetForBEVDet, self).__init__()
        #build backbone
        # assert len(num_layer)>=3
        assert len(num_layer)==len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[Bottleneck(curr_numC, num_channels[i]//4, stride=stride[i],
                                downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([Bottleneck(curr_numC, curr_numC//4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[BasicBlock(curr_numC, num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        if self.neck is not None:
            feats = self.neck(feats)
        # support centerhead inputs
        # feats = [feats]
        return feats

@NECKS.register_module()
class BEVDepth3DNeck(BaseModule):
    def __init__(self,
                 bev_backbone,
                 bev_neck):
        super().__init__()
        self.backbone = build_backbone(bev_backbone)
        self.neck = build_neck(bev_neck)

    def forward(self, x):
        trunk_outs = [x]
        
        if self.backbone.deep_stem:
            x = self.backbone.stem(x)
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.norm1(x)
            x = self.backbone.relu(x)
        for i, layer_name in enumerate(self.backbone.res_layers):
            res_layer = getattr(self.backbone, layer_name)
            x = res_layer(x)
            if i in self.backbone.out_indices:
                trunk_outs.append(x)
        fpn_output = self.neck(trunk_outs)[0]
        return fpn_output




@NECKS.register_module()
class ResNetForBEVDetv2(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
            neck=None,
    ):
        super(ResNetForBEVDetv2, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats
