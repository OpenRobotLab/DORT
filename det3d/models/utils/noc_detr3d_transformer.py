import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER

from .detr3d_transformer import Detr3DTransformer, Detr3DTransformerDecoder, inverse_sigmoid, nan_to_num
from mmdet3d.core.bbox.coders import build_bbox_coder
from det3d.core.bbox.util import rot_yaw_matrix


@TRANSFORMER.register_module()
class NocDetr3DTransformer(Detr3DTransformer):
    """Implement the Noc_based Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            'as_two_stage' as True; Default: 300.
    """

    def __init__(self,
                num_feature_levels=4,
                num_cams=6,
                two_stage_num_proposals=300,
                decoder=None,
                code_size=8,
                num_reference_points=4,
                **kwargs):
        self.num_reference_points = num_reference_points
        self.code_size = code_size
        super().__init__(num_feature_levels,
                        num_cams,
                        two_stage_num_proposals,
                        decoder,
                        **kwargs)


    def init_layers(self):
    #     """ Initialize layers of the NocDetr3DTransformer."""
    #     # TODO add the per layer head?
    #     self.reference_points_branch = nn.Linear(self.embed_dims, self.num_reference_points * 3)
        self.init_box_branch = nn.Linear(self.embed_dims, 8)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, DeformableDetr3DCrossAtten):
                m.init_weight()
        xavier_init(self.init_box_branch, distribution='uniform', bias=0.)


    def forward(self,
                mlvl_feats,
                query_embed,
                reg_branchs=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims , dim=1)
        # TODO redefine the query embedding -> can based on 2D box roi-aligned features
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        # TODO redefine the query pos -> query pos is used for generated init box
        # can be initialized from fov detector -> merged with nms
        query = query.unsqueeze(0).expand(bs, -1, -1)


        # reference_points = self.reference_points(query_pos)
        # reference_points = reference_points.sigmoid()
        # init_box = reference_points

        # decoder
        query = query.permute(1, 0, 2) # query; bs, -1
        query_pos = query_pos.permute(1, 0, 2) # query; bs; -1

        init_boxes = self.init_box_branch(query_pos)  # query; bs -1

        inter_states, inter_boxes = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            init_boxes=init_boxes,
            reg_branchs=reg_branchs,
            **kwargs)

        return inter_states, init_boxes.permute(1, 0, 2), inter_boxes


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class NocDetr3DTransfromerDecoder(TransformerLayerSequence):
    """Implements the decoder in NocDETR3D transformer.
    Args:
        return_intermediate (bool): whether to return intermediate outputs.
        coord_norm_cfg (dict): Config of last normalization layer. Default:
            'LN'.

    """

    def __init__(self, *args,
                    num_reference_points=8,
                    return_intermediate=False,
                    detach_boxes=True,
                    code_size=8,
                    pc_range=None,
                    bbox_coder=None,
                    encode_dim=True,
                    encode_rot=True,
                    **kwargs):
        super(NocDetr3DTransfromerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_reference_points = num_reference_points
        self.detach_boxes = detach_boxes
        self.code_size=code_size
        self.pc_range=pc_range
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.encode_dim=encode_dim
        self.encode_rot=encode_rot
        # self.reference_points_branch =  nn.Linear(self.embed_dims, self.num_reference_points * 3)
        # TODO check if I need to call the init layers
        self.init_layers()

    def init_layers(self):
        # pass
        self.reference_points_branch =  nn.Linear(self.embed_dims, self.num_reference_points * 3)


    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, Detr3DCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points_branch, distribution='uniform', bias=0.)


    def forward(self,
                query,
                *args,
                init_boxes=None,
                reg_branches=None,
                **kwargs):

        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = query
        intermediate = []
        intermediate_boxes = []
        num_query, bs, embed_dim = query.shape
        # boxes = init_boxes
        tmp = reg_branches[0](query)
        boxes = tmp
        boxes[..., 0:2] = init_boxes[..., 0:2]
        boxes[..., 4:5] = init_boxes[..., 4:5]

        for lid, layer in enumerate(self.layers):
            # boxes_input = boxes
            # generated nocs
            reference_points_noc = self.reference_points_branch(output).tanh()
            # reference_points_noc = boxes.new_zeros(num_query, bs, self.num_reference_points, 3)
            # noc
            reference_points_noc = reference_points_noc.reshape( num_query, bs, self.num_reference_points, 3)

            reference_points_3d, boxes_decoded = points_noc_to_3d(
                    reference_points_noc,
                    boxes,
                    bbox_coder=self.bbox_coder,
                    encode_dim=self.encode_dim,
                    encode_rot=self.encode_rot)
            output = layer(
                output,
                *args,
                boxes=boxes,
                boxes_decoded=boxes_decoded,
                reference_points_noc = reference_points_noc,
                reference_points_3d = reference_points_3d,
                **kwargs)
            # output = output.permute(1, 0, 2)
            if reg_branches is not None:
                # check how to update all the boxes
                tmp = reg_branches[lid](output)


                # boxes[..., :self.code_size] += tmp[..., :self.code_size]
                new_boxes = torch.zeros_like(boxes)
                # new_boxes[..., :6] += tmp[...,:6]
                new_boxes[..., :2] = tmp[..., :2] + boxes[..., :2]
                new_boxes[..., 4:5] = tmp[...,4:5] + boxes[...,4:5]

                boxes = new_boxes
                if self.detach_boxes:
                    boxes = boxes.detach()

                # new_reference_points = torch.zeros_like(reference_points)
                # new_reference_points[..., :2] = tmp[
                #     ..., :2] + inverse_sigmoid(reference_points[..., :2])
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                # new_reference_points = new_reference_points.sigmoid()

                # reference_points = new_reference_points.detach()
                # boxes = self.
            # output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_boxes.append(boxes.permute(1, 0, 2)) # to bs; query; -1

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_boxes)
        return output, boxes

def points_noc_to_3d(noc_points,
                    boxes, bbox_coder=None,
                    encode_dim=True, encode_rot=True, ):
    '''
    Input:
        Args:
            nocs_points; torch.array with shape n_query x bsz x num_points x 3
            boxes; torch.array with shape n_query x bsz x 8
            bbox_coder; nms free bbox coder used for denormalized bbox
            encode_dim; encode bbox dim in nocs?
            encode_rot; encode bbox rot in nocs?
    Output:
        nocs_points in 3d coordinate -> torch.array
    '''
    # 1. denormalize box
    boxes = boxes.clone()
    boxes = boxes.permute(1, 0, 2) #bs query -1
    boxes[...,0:2] = boxes[...,0:2].sigmoid()
    boxes[...,4:5] = boxes[...,4:5].sigmoid()
    pc_range = bbox_coder.pc_range

    boxes[...,0:1] = (boxes[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
    boxes[...,1:2] = (boxes[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    boxes[...,4:5] = (boxes[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2])
    boxes = bbox_coder.decode_box(boxes)
    boxes = boxes.permute(1, 0, 2)
    boxes_decoded = boxes.clone() # query; bs; -1

    num_query, bsz, num_points, _ = noc_points.shape
    # 2. covert the noc points to 3D coordinate
    if encode_dim:
        dense_dims = torch.cat([boxes[..., 2:4],
                            boxes[..., 5:6]], dim=-1)

        # 3. convert based on dimension

        dense_dims = dense_dims.unsqueeze(2).expand(-1, -1, num_points, -1)
        noc_points = noc_points * dense_dims


    # 4. convert based on rotation

    if encode_rot:
        yaw = boxes[..., 6:7]
        rot_matrix = rot_yaw_matrix(yaw.reshape(-1))
        rot_matrix = rot_matrix.reshape(num_query, bsz, 1, 3, 3)
        # rot_matrix = rot_matrix.expand()
        noc_points = torch.matmul(noc_points.unsqueeze(-2), rot_matrix).squeeze(-2)

    # 5. fill with 3D location
    dense_loc = torch.cat([boxes_decoded[...,0:2],
                        boxes_decoded[..., 4:5]], dim=-1)

    dense_loc = dense_loc.unsqueeze(2).expand(-1, -1, num_points, -1)
    noc_points = noc_points + dense_loc



    return noc_points, boxes_decoded


@ATTENTION.register_module()
class DeformableDetr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                embed_dims=256,
                num_heads=8,
                num_levels=4,
                num_points=5,
                num_cams=6,
                im2col_step=64,
                pc_range=None,
                dropout=0.1,
                norm_cfg=None,
                init_cfg=None,
                code_size=8,
                attn_way="average",
                batch_first=False):
        super(DeformableDetr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.code_size=code_size


        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # TODO support batch first
        self.attn_way = attn_way

        if self.attn_way == "average":
            pass
        elif self.attn_way == "product":
            self.attn_module = nn.MultiheadAttention(embed_dims, num_heads=1)
        else:
            raise NotImplementedError


        self.position_box_encoder = nn.Sequential(
            nn.Linear(7, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.position_noc_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)


    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points_noc=None,
                reference_points_3d=None,
                boxes=None,
                boxes_decoded=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 3),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        # from num_query x bsz x num_points x 3 to bsz x num_query x num_points x 3
        reference_points_3d = reference_points_3d.permute(1, 0, 2, 3)
        reference_points_noc = reference_points_noc.permute(1, 0, 2, 3)
        # get the reference_points pos encoding
        reference_points_pos = self.position_noc_encoder(reference_points_noc.reshape(-1, 3))
        # reference_points_pos = reference_points_pos.
        reference_points_3d, output, mask = deformable_feature_sampling(
            value, reference_points_3d, self.pc_range, kwargs['img_metas'])

        B, C, num_query, num_points, num_cam, num_level = output.shape

        output_attn = output.permute(0, 4, 5, 2, 3, 1).reshape(-1, num_points, C)
        # output_attn = output_attn +

        reference_points_pos = reference_points_pos.view(B, 1, 1, num_query, num_points, C)
        reference_points_pos = reference_points_pos.expand(-1, num_cam, num_level, -1, -1, -1)
        output_attn += reference_points_pos.reshape(-1, num_points, C)
        # # output =

        if self.attn_way == "average":
            output_attn = output_attn.mean(1, keepdim=True)

        elif self.attn_way == "product":
            query_attn = query.unsqueeze(0).expand(num_cam*num_level, -1, -1, -1).reshape(-1, 1, C)
            # to sequence x batch x feature dim
            query_attn = query_attn.permute(1, 0, 2)
            output_attn = output_attn.permute(1, 0, 2)
            # output_pos =
            output_attn = self.attn_module(query_attn, output_attn, output_attn)[0]
            output_attn = output_attn.permute(1, 0, 2)
        else:
            raise NotImplementedError

            
        # output_attn = output_attn.mean(1, keepdim=True)
        output = output_attn.reshape(B, num_cam, num_level, num_query, C).permute(0, 4, 3, 1, 2)
        output = output.unsqueeze(-2)
        # do the cross-attention

        # add the position encoding from boxes and nocs.

        output = nan_to_num(output)
        #mask = nan_to_num(mask)
         # what is mask
        attention_weights = attention_weights.sigmoid() #* mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        # boxes_loc = torch.cat([boxes[..., 0:2], boxes[...,4:5]], dim=-1)
        # import pdb; pdb.set_trace()
        pos_feat = self.position_box_encoder(boxes[...,:7])
        # pos_feat = pos_feat.

        return self.dropout(output) + inp_residual + pos_feat


# def query_wise_feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
#     pass

# TODO reconfiguration with feature sampling in detr3d
def deformable_feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query, num_points = reference_points.size()[:3]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, num_points, 4)
    reference_points = reference_points.repeat(1, num_cam, 1, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 1, 4, 4).repeat(1, 1, num_query, num_points, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    reference_points_cam = reference_points_cam.view(B, num_cam, num_query*num_points, -1)
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.max(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= img_metas[0]['pad_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['pad_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query*num_points, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query*num_points, 1, 2)
        reference_points_cam_lvl = reference_points_cam_lvl.to(feat.dtype)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl, padding_mode="zeros")
        sampled_feat = sampled_feat.view(B, N, C, num_query*num_points, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    # DO the attention for num_points ->  1xn ->
    # sampled_feats = sampled_feats.clone()
    mask = mask.expand(-1, C, -1, -1, -1, len(mlvl_feats)).detach()
    sampled_feats = sampled_feats * mask.float()

    sampled_feats = sampled_feats.view(B, C, num_query, num_points, num_cam, len(mlvl_feats))

    # add the mask
    return reference_points_3d, sampled_feats, mask
