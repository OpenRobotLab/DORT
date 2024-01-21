import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


def rot_yaw_matrix(yaw):
    device = yaw.device
    N = yaw.shape[0]
    cos, sin = yaw.cos(), yaw.sin()
    i_temp = yaw.new_tensor([[1, 0, 1],
                            [0, 1, 0],
                            [-1, 0, 1]])
    ry = i_temp.repeat(N, 1).view(N, -1, 3)
    ry[:, 0, 0] *= cos
    ry[:, 0, 2] *= sin
    ry[:, 2, 0] *= sin
    ry[:, 2, 2] *= cos

    return ry

@BBOX_CODERS.register_module()
class NOCSCoder(BaseBBoxCoder):
    """Coder for normalized object coordinate
    """

    def __init__(self, with_dim=True, with_yaw=True):

        self.with_dim = with_dim
        self.with_yaw = with_yaw


    def decode(self, nocs, dimension, yaw):
        if len(nocs) == 0:
            return nocs
        coord_shape = nocs.shape

        if nocs.dim() > 2:
            nocs = nocs.permute(0, 2, 3, 1).reshape(-1, 3)

            yaw = yaw.permute(0, 2, 3, 1).reshape(-1)

            dimension = dimension.permute(0, 2, 3, 1).reshape(-1, 3)

        if self.with_dim:
            nocs = nocs * dimension.clone()

        rot_matrix = rot_yaw_matrix(yaw.reshape(-1))
        if self.with_yaw:
            nocs = torch.bmm(rot_matrix.float(), nocs.unsqueeze(-1))
        nocs = nocs.squeeze(-1)

        if len(coord_shape) > 2:
            nocs = nocs.reshape(
                coord_shape[0],
                coord_shape[2],
                coord_shape[3],
                coord_shape[1]).permute(0, 3, 1, 2)
        return nocs
    def encode(self, nocs, dimension, yaw):
        raise NotImplementedError
