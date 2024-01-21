import mmcv
import numpy as np
import trimesh
from os import path as osp
import copy
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmdet3d.core.bbox.structures import Box3DMode
# from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img,
#                                               draw_camera_bbox3d_on_img,
#                                               draw_depth_bbox3d_on_img

def show_custom_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               out_dir,
                               filename,
                               box_mode='lidar',
                               img_metas=None,
                               show=False,
                               gt_bbox_color=(61, 102, 255),
                               pred_bbox_color=(241, 101, 72)):
    """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (list of np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str, optional): Coordinate system the boxes are in.
            Should be one of 'depth', 'lidar' and 'camera'.
            Defaults to 'lidar'.
        img_metas (dict, optional): Used in projecting depth bbox.
            Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int), optional): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61).
        pred_bbox_color (str or tuple(int), optional): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241).
    """
    draw_bbox = draw_custom_lidar_bbox3d_on_img

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        raise NotImplementedError
        show_img = img.copy()
        if gt_bboxes is not None:
            show_img = draw_bbox(
                gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
        if pred_bboxes is not None:
            show_img = draw_bbox(
                pred_bboxes,
                show_img,
                proj_mat,
                img_metas,
                color=pred_bbox_color)
        mmcv.imshow(show_img, win_name='project_bbox3d_img', wait_time=0)
    for idx, img_idx in enumerate(img):
        img_idx = img_idx.transpose(1, 2, 0)
        if img is not None:
            mmcv.imwrite(img_idx, osp.join(result_path, f'{filename}_img_{idx}.png'))

        lidar2cam_idx = img_metas["lidar2cam"][idx]
        cam2img_idx = img_metas["cam2img"][idx]
        if gt_bboxes is not None:
            gt_img = draw_bbox(
                gt_bboxes, img_idx, lidar2cam_idx, cam2img_idx, img_metas, color=gt_bbox_color)
            mmcv.imwrite(gt_img, osp.join(result_path, f'{filename}_gt_{idx}.png'))

        if pred_bboxes is not None:
            pred_img = draw_bbox(
                pred_bboxes, img_idx, lidar2cam_idx, cam2img_idx, img_metas, color=pred_bbox_color)
            mmcv.imwrite(pred_img, osp.join(result_path, f'{filename}_pred_{idx}.png'))



def draw_custom_lidar_bbox3d_on_img(bboxes3d,
                                    raw_img,
                                    lidar2cam_rt,
                                    cam2img_rt,
                                    img_metas,
                                    color=(0, 255, 0),
                                    thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (numpy.array, shape=[M, 7]):
            3d bbox (x, y, z, d)
    Args:
        bboxes3d (numpy.array, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (deprecated) (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    extrinsic = lidar2cam_rt
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.cpu().numpy()
    bboxes3d = bboxes3d.convert_to(
        Box3DMode.CAM, rt_mat=extrinsic)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    #
    intrinsic = cam2img_rt
    # lidar2img_rt = copy.deepcopy(np.dot(intrinsic, extrinsic.T)).reshape(4,4)
    # lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    # if isinstance(lidar2img_rt, torch.Tensor):
        # lidar2img_rt = lidar2img_rt.cpu().numpy()
    # import pdb; pdb.set_trace
    # pts_2d = pts_4d @ extrinsic.T
    center = bboxes3d.gravity_center
    center_2d = np.concatenate(
        [center.reshape(-1, 3),
         np.ones((num_bbox, 1))], axis=-1)
    center_2d = center_2d @ intrinsic.T
    pts_2d = pts_4d @ intrinsic.T
    mask = pts_2d[:, 2] <= 1
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_bbox):
        corners = imgfov_pts_2d[i].astype(np.int)
        if mask[i] is True:
            continue
        centers = np.mean(corners, axis=0)
        # check centers
        flag = check_centers(centers, img.shape)
        if flag is False:
            continue
        corners[:,0] = np.clip(corners[:,0], a_min=0, a_max=img.shape[1] - 1)
        corners[:,1] = np.clip(corners[:,1], a_min=0, a_max=img.shape[0] - 1)


        # do the clamp
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)



def check_centers(centers, shape):
    if np.min(centers) < 0:
        return False
    if centers[0] > shape[1]:
        return False
    if centers[1] > shape[0]:
        return False

    return True
