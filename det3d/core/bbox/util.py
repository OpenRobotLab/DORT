import torch
from mmdet3d.core.bbox.structures.utils import points_cam2img
import math
import numpy as np

def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
         # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

def alpha_to_ry(location, alpha):
    ray = torch.atan2(location[:, 2], location[:, 0])
    ry = alpha + (-ray)
    ry = ry + 0.5 * math.pi
    ry = (ry + math.pi) % (2 * math.pi) - math.pi
    return ry

def custom_points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points from camera coordicates to image coordinates.

    Args:
        points_3d (torch.Tensor): Points in shape (N, 3).
        proj_mat (torch.Tensor): Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        torch.Tensor: Points in image coordinates with shape [N, 2].
    """
    points_num = list(points_3d.shape)[:-1]

    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yeilds better results
    points_4 = torch.cat(
        [points_3d, points_3d.new_ones(*points_shape)], dim=-1)
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / (point_2d[..., 2:3].abs() + 1e-4)

    if with_depth:
        return torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res


def projected_gravity_center(bbox, rt_mat):
    gravity_center = bbox.gravity_center
    projected_gravity_center = custom_points_cam2img(gravity_center, rt_mat.float())
    return projected_gravity_center

def projected_2d_box(bbox, rt_mat, img_shape):

    corners = bbox.corners
    corners_2d = points_cam2img(corners, rt_mat.float())
    corners_2d[..., 0].clamp_(min=0, max=img_shape[1])
    corners_2d[..., 1].clamp_(min=0, max=img_shape[0])

    corners_2d = torch.stack([
        corners_2d[..., 0].min(dim=-1)[0],
        corners_2d[..., 1].min(dim=-1)[0],
        corners_2d[..., 0].max(dim=-1)[0],
        corners_2d[..., 1].max(dim=-1)[0],
    ], dim=-1)
    return corners_2d
    # corners_2d



def points_cam2img_batch(points_3d, proj_mat):
    """
    projects points from camera coordinates to image coordiantes with multiple camera
    """
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    # previous implementation use new_zeros, new_one yeilds better results
    points_4 = torch.cat(
        [points_3d, points_3d.new_ones(*points_shape)], dim=-1)
    points_4 = points_4.unsqueeze(1)
    point_2d = torch.bmm(points_4, proj_mat.permute(0, 2, 1))
    point_2d = point_2d.squeeze(1)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def points_cam2img_broadcast(points_3d, proj_mat):
    """
    projects points from camera coordinates to image coordiantes with multiple camera
    """
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    # previous implementation use new_zeros, new_one yeilds better results
    points_4 = torch.cat(
        [points_3d, points_3d.new_ones(*points_shape)], dim=-1)
    points_4 = points_4.unsqueeze(-2)
    point_2d = torch.matmul(points_4, proj_mat)
    point_2d = point_2d.squeeze(-2)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res



# def points_img2cam(points_2d, depth, proj_mat):
#     """Lift points from image coordinate to camera coordinates.
#     Args:
#         points_2d (torch.Tensor): Points in shape (N, 2)
#         depth (torch.Tensor): points in shape (N, 1)
#         proj_mat (torch.Tensor): Transformation matrix between coordinate
#     Returns:
#         torch.Tensor: Points in camera coordinate with shape [N, 3]
#     """
#     # TODO convert it to directly inverse
#     # assert p
#     points_num = list(points_2d.shape)[:-1]
#     # points_3 = torch.cat([points_2d, depth], dim=-1)
#     # points_3[:,:2] *= depth

#     # points_3d = torch.matmul(torch.inverse(proj_mat)[:3,:3], points_3.T)
#     # points+shape = np.concatenate([points_num, ])
#     z = depth + proj_mat[2,3]
#     x = (points_2d[:,0:1] * z - proj_mat[0,3] - proj_mat[0,2] * depth)  / proj_mat[0,0]
#     y = (points_2d[:,1:2] * z - proj_mat[1,3] - proj_mat[1,2] * depth)  / proj_mat[1,1]
#     return torch.cat([x, y, z], dim=-1)



def points_img2cam(points_2d, depth, proj_mat):
    """Lift points from image coordinate to camera coordinates.
    Args:
        points_2d (torch.Tensor): Points in shape (N, M 2)
        depth (torch.Tensor): points in shape (N, 1)
        proj_mat (torch.Tensor): Transformation matrix between coordinate
    Returns:
        torch.Tensor: Points in camera coordinate with shape [N, 3]
    """
    # TODO convert it to directly inverse
    # assert p
    points_num = list(points_2d.shape)[:-1]
    # points_3 = torch.cat([points_2d, depth], dim=-1)
    # points_3[:,:2] *= depth

    # points_3d = torch.matmul(torch.inverse(proj_mat)[:3,:3], points_3.T)
    # points+shape = np.concatenate([points_num, ])

    z = depth + proj_mat[2,3]
    x = (points_2d[:,0:1] * z - proj_mat[0,3] - proj_mat[0,2] * depth)  / proj_mat[0,0]
    y = (points_2d[:,1:2] * z - proj_mat[1,3] - proj_mat[1,2] * depth)  / proj_mat[1,1]
    return torch.cat([x, y, z], dim=-1)

def points_img2cam_batch(points_2d, depth, proj_mat):
    """Lift points from image coordinate to camera coordinates.
    Args:
        points_2d (torch.Tensor): Points in shape (N 2)
        depth (torch.Tensor): points in shape (N, 1)
        proj_mat (torch.Tensor): Transformation matrix between coordinate (N, 4,4)
    Returns:
        torch.Tensor: Points in camera coordinate with shape [N, 3]
    """
    # TODO convert it to directly inverse
    # assert p
    # depth =
    points_num = list(points_2d.shape)[:-1]
    # points_3 = torch.cat([points_2d, depth], dim=-1)
    # points_3[:,:2] *= depth

    # points_3d = torch.matmul(torch.inverse(proj_mat)[:3,:3], points_3.T)
    # points+shape = np.concatenate([points_num, ])

    z = depth[:,0] + proj_mat[:, 2,3]
    x = (points_2d[:, 0] * z - proj_mat[:,0,3] - proj_mat[:,0,2] * depth[:,0])  / proj_mat[:,0,0]
    y = (points_2d[:,1] * z - proj_mat[:,1,3] - proj_mat[:,1,2] * depth[:,0])  / proj_mat[:,1,1]
    return torch.stack([x, y, z], dim=-1)


def bbox_alpha(bbox):
    roty = bbox.yaw
    location = bbox.center
    ray = torch.atan2(location[:, 2], location[:,0])
    alpha = roty - (-ray)
    pi = math.pi
    alpha = alpha - 0.5 * pi
    alpha = (alpha + pi) % (2 * pi) - pi
    return alpha



def unnormalized_coordinate(object_coordinate, dimension, yaw):
    if len(object_coordinate) == 0:
        return object_coordinate
    coord_shape = object_coordinate.shape

    if object_coordinate.dim() > 2:
        object_coordinate = object_coordinate.permute(0, 2, 3, 1).reshape(-1, 3)

        yaw = yaw.permute(0, 2, 3, 1).reshape(-1)

        dimension = dimension.permute(0, 2, 3, 1).reshape(-1, 3)


    object_coordinate = object_coordinate * dimension.clone()

    rot_matrix = rot_yaw_matrix(yaw.reshape(-1))
    object_coordinate = torch.bmm(rot_matrix.float(), object_coordinate.unsqueeze(-1))
    object_coordinate = object_coordinate.squeeze(-1)

    if len(coord_shape) > 2:
        object_coordinate = object_coordinate.reshape(
            coord_shape[0],
            coord_shape[2],
            coord_shape[3],
            coord_shape[1]).permute(0, 3, 1, 2)
    return object_coordinate

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



def centernet_distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """
    center_x = points[..., 0] + distance[..., 0]
    center_y = points[..., 1] + distance[..., 1]
    # dim_x = distance[..., 2]
    x1 = center_x - distance[..., 2]/2.
    x2 = center_x + distance[..., 2]/2.
    y1 = center_y - distance[..., 3]/2.
    y2 = center_y + distance[..., 3]/2.

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            from mmdet.core.export import dynamic_clip_for_onnx
            x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes



# def normalized_coordinate(object_coordinate, dimension, yaw):
#     if len(object_coordinate) == 0:
#         return object_coordinate
#     coord_shape = object_coordinate.shape

#     if object_coordinate.dim() > 2:
#         object_coordinate = object_coordinate.permute(0, 2, 3, 1).reshape(-1, 3)
#         yaw = yaw.permute(0, 2, 3, 1).reshape(-1)

#         dimension = dimension.permute(0, 2, 3, 1).reshape(-1, 3)

#     rot_matrix = rot_yaw_matrix(yaw.reshape(-1))
#     rot_matrix = rot_matrix.permute(0, 2, 1)
#     object_coordinate = torch.bmm(rot_matrix.float(), object_coordinate.unsqueeze(-1))
#     object_coordinate = object_coordinate * dimension
#     object_coordinate = object_coordinate.squeeze(-1)
#     if len(coord_shape) > 2:
#         object_coordinate = object_coordinate.reshape(
#             coord_shape[0],
#             coord_shape[2],
#             coord_shape[3],
#             coord_shape[1]).permute(0, 3, 1, 2)
#     return object_coordinat


