import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
import torch


import numpy as np
def mmdet3d_to_nusc_bbox(bboxes):
    box_gravity_center = bboxes.gravity_center.numpy()
    box_dims = bboxes.dims.numpy()
    box_yaw = bboxes.yaw.numpy()
    nus_box_dims = box_dims[:, [1, 0, 2]]
    box_list = []
    for i in range(len(bboxes)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if bboxes.tensor.shape[1] > 7:
            velocity = (*bboxes.tensor[i, 7:9], 0.0)
        else:
            velocity = (0.0, 0.0, 0.0)
        # velo_val = np.linalg.norm(bboxes[i, 7:9])
        # velo_ori = bboxes[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=0,
            score=1,
            velocity=velocity)
        box_list.append(box)
    return box_list

def nusc_to_mmdet3d_bbox(bboxes):
    locs = np.array([b.center for b in bboxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in bboxes]).reshape(-1, 3)
    # print(dims)
    dims = dims[:, [1, 0, 2]]
    rots = np.array([b.orientation.yaw_pitch_roll[0] \
                     for b in bboxes]).reshape(-1, 1)
    
    bboxes = np.concatenate([locs, dims, rots], axis=1)
    bboxes = torch.tensor(bboxes)
    bboxes = LiDARInstance3DBoxes(bboxes)
    return bboxes

def lidar_nusc_bbox_to_global(boxes, img_metas):
    box_list = []
    for box in boxes:
        box.rotate(pyquaternion.Quaternion(img_metas['lidar2ego_rotation']))
        box.translate(np.array(img_metas['lidar2ego_translation']))
        box.rotate(pyquaternion.Quaternion(img_metas['ego2global_rotation']))
        box.translate(np.array(img_metas['ego2global_translation']))
        box_list.append(box)
    return box_list