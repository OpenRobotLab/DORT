import numpy as np
import torch
import copy
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Resize
from mmdet3d.core.bbox.structures import Box3DMode
from mmdet3d.datasets.pipelines import RandomFlip3D, Compose
from mmdet.datasets.pipelines import RandomFlip, RandomCrop
from det3d.core.bbox.util import projected_2d_box
from mmdet3d.core.bbox.box_np_ops import points_in_rbbox
# from mmdet3d.d

from mmdet3d.datasets.pipelines import LoadAnnotations3D
from mmdet3d.datasets.pipelines import ObjectRangeFilter
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes,)

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""

        results['img_shape'] = [img.shape for img in results['img']] # should be pad shape

        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img'] = padded_img
        # results['img_shape'] = [img.shape for img in padded_img] # should be pad shape

        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg_mask(self, key, results):
        if self.size is not None:
            padded_mask = [mmcv.impad(
                mask, shape=self.size, pad_val=self.pad_val) for mask in results[key]]
        elif self.size_divisor is not None:
            padded_mask = [mmcv.impad_to_multiple(
                mask, self.size_divisor, pad_val=self.pad_val) for mask in results[key]]
        results[key] = padded_mask
        return results

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)

        # seg_fields = results.get('seg_fields', None)
        # if seg_fields is not None:
        #     for key in seg_fields:
        #         results = self._pad_seg_mask(key, results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class CropMultiViewImage(object):
    """Crop the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, size=None):
        self.size = size

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        results['img'] = [img[:self.size[0], :self.size[1], ...] for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img_fixed_size'] = self.size
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        return repr_str


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    # NOTE this is implemented by detr3d
    """Random scale the image
    Args:
        scales
        TODO consider how to improve it
        # different level of augmentation (from the bev space of the fov space).
    """

    def __init__(self, scales=[0.5, 1.0, 1.5]):
        self.scales = scales

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        np.random.shuffle(self.scales)
        rand_scale = self.scales[0]
        img_shape = results['img_shape'][0]
        y_size = int(img_shape[0] * rand_scale)
        x_size = int(img_shape[1] * rand_scale)
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size, y_size), return_scale=False) for img in results['img']]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        results['gt_bboxes_3d'].tensor[:, :6] *= rand_scale
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str

@PIPELINES.register_module()
class RandomScaleImage3D(Resize):
    """Random scale the image with modifying camera intrinsic
    Args:
        img_scale: resolution for the output image
        keep_ratio: keep the ratio between w and h
        resize_depth: consider simltaneously resize depth (for training aug)
        ratio_range:
        override: do the resize more than one time.
    """
    def __init__(self, img_scale=None,
                         multiscale_mode='range',
                         ratio_range=None,
                         keep_ratio=True,
                         bbox_clip_border=True,
                         backend='cv2',
                         resize_depth=False,
                         rescale_intrinsic=True,
                         override=False):
        '''
            if resize depth is False, this is equal to
                vannila augmentation and only consider recale cam intrinsic
            if resize depth is True,
                we further consider resize depth by the corresponding ratio.
        '''
        self.resize_depth=resize_depth

        super(RandomScaleImage3D, self).__init__(img_scale=img_scale,
                                    multiscale_mode=multiscale_mode,
                                    ratio_range=ratio_range,
                                    keep_ratio=keep_ratio,
                                    bbox_clip_border=bbox_clip_border,
                                    backend=backend, override=override)
        self.rescale_intrinsic = rescale_intrinsic


    def scale_centers2d(self, results):
        ratio = self.get_ratio(results)
        if "centers2d" in results:
            results['centers2d'] *= ratio

        return results

    def scale_intrinsic(self, results):
        if 'cam2img' in results:
            ratio = self.get_ratio(results)

            for idx, cam2img_idx in enumerate(results['cam2img']):
                cam2img_idx*= ratio
                cam2img_idx[2,2] = 1
                cam2img_idx[3,3] = 1
                results['cam2img'][idx] = cam2img_idx
                results['lidar2img'][idx] = cam2img_idx @ results['lidar2cam'][idx]

        return results

    def scale_depth(self, results):
        results['depth_ratio'] = results['img_shape'][0][0] / self.img_scale[0][1]
        return results

    def __call__(self, results):
        # self.drop_ratio = drop_ratio
        # super(RandomScaleImage3D, self).__call__(input_dict)
        # Assume the multiview image is in the same shape
        # currently img is with the shape of [N, C, H, W, M]

        # NOTE!!!! i remove the check scale_factor part due to load multiview images
        if 'scale' not in results:
            if 'scale_factor' in results:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)

        results = self.scale_centers2d(results)
        if self.rescale_intrinsic:
            results = self.scale_intrinsic(results)
        if self.resize_depth:
            results = self.scale_depth(results)
        return results


    def get_ratio(self, results):
        ratio = results['img_shape'][0] / results['ori_shape'][0]
        return ratio

    def _resize_img(self, results):
        """Resize images with list of inputs
        ``results['scale']``."""

        # other with type of tensor
        for idx, img_idx in enumerate(results['img']):
            if self.keep_ratio:
                resized_img_idx, scale_factor = mmcv.imrescale(
                    img_idx,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = resized_img_idx.shape[:2]
                h, w = img_idx.shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                resized_img_idx, w_scale, h_scale = mmcv.imresize(
                    img_idx,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results['img'][idx] = resized_img_idx

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)

        results['img_shape'] = results['img'][0].shape
        # in case that there is no padding
        results['pad_shape'] = results['img'][0].shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

@PIPELINES.register_module()
class HorizontalRandomFlipMultiViewImage(object):

    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = 0.5

    def __call__(self, results):
        if np.random.rand() >= self.flip_ratio:
            return results
        results = self.flip_bbox(results)
        results = self.flip_cam_params(results)
        results = self.flip_img(results)
        return results

    def flip_img(self, results, direction='horizontal'):
        results['img'] = [mmcv.imflip(img, direction) for img in results['img']]
        return results

    def flip_cam_params(self, results):
        flip_factor = np.eye(4)
        flip_factor[1, 1] = -1
        lidar2cam = [l2c @ flip_factor for l2c in results['lidar2cam']]
        w = results['img_shape'][0][1]
        lidar2img = []
        for cam2img, l2c in zip(results['cam2img'], lidar2cam):
            cam2img[0, 2] = w - cam2img[0, 2]
            lidar2img.append(cam2img @ l2c)
        results['lidar2cam'] = lidar2cam
        results['lidar2img'] = lidar2img
        return results

    def flip_bbox(self, input_dict, direction='horizontal'):
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)
        return input_dict



@PIPELINES.register_module()
class CustomRandomFlip3D(RandomFlip3D):


    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            results['img'] = [mmcv.imflip(img, results['flip_direction']) for img in results['img']]
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])

        if self.sync_2d:
            results['pcd_horizontal_flip'] = results['flip']
            results['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in results:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                results['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in results:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                results['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in results:
            results['transformation_3d_flow'] = []

        if results['pcd_horizontal_flip']:
            self.random_flip_data_3d(results, 'horizontal')
            results['transformation_3d_flow'].extend(['HF'])
        if results['pcd_vertical_flip']:
            self.random_flip_data_3d(results, 'vertical')
            results['transformation_3d_flow'].extend(['VF'])
        return results

    def flip_img(self, results, direction='horizontal'):
        results['img'] = [mmcv.imflip(img, direction) for img in results['img']]
        return results


@PIPELINES.register_module()
class CustomRandomFlip3Dv2(object):
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio=flip_ratio

    def __call__(self, results):
        if np.random.rand() > self.flip_ratio:
            return results
        results = self.flip_img(results)
        results = self.flip_bbox_cam_params(results)
        return results

    def flip_img(self, results, direction='horizontal'):
        results['img'] = [mmcv.imflip(img, direction) for img in results['img']]
        return results



    def flip_bbox_cam_params(self, results):
        if "gt_bboxes_3d" not in results:
            return results

        assert len(results["cam2img"]) == 1
        lidar2cam_idx = results['lidar2cam'][0]
        cam2img_idx = results['cam2img'][0]

        box_cam =results['gt_bboxes_3d'].convert_to(Box3DMode.CAM, rt_mat=lidar2cam_idx)
        d = box_cam.tensor[:,2]
        x = box_cam.tensor[:,0]
        w = results['img_shape'][1]
        f = cam2img_idx[0, 0]
        u = cam2img_idx[0, 2]
        T_x = cam2img_idx[0, 3]
        x = (d * w - 2 * u * d - 2 * T_x - f * x) / f
        rot_y = box_cam.yaw
        pos_mask = rot_y > 0
        rot_y[pos_mask] = np.pi - rot_y[pos_mask]
        rot_y[~pos_mask] = - np.pi - rot_y[~pos_mask]
        box_cam.tensor[:, 0] = x
        box_cam.tensor[:, 6] = rot_y
        box_lidar = box_cam.convert_to(Box3DMode.LIDAR, rt_mat=np.linalg.inv(lidar2cam_idx))
        results['gt_bboxes_3d'] = box_lidar
        return results



#deprecated
# @PIPELINES.register_module()
# class KittiSetOrigin:
#     def __init__(self, point_cloud_range):
#         point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
#         self.origin = (point_cloud_range[:3] + point_cloud_range[3:]) / 2.

#     def __call__(self, results):
#         if isinstance(results['lidar2img'], list):
#             for idx in range(len(results['lidar2img'])):
#                 results['lidar2img'][idx]['origin'] = self.origin.copy()
#         else:
#             results['lidar2img']['origin'] = self.origin.copy()
#         return results



#deprecated
# @PIPELINES.register_module()
# class FilterCornerBackCam:
#     # Filter the bounding boxes that their corner depth is < 0
#     # currently this is designed for monocular data.
#     def __init__(self, filter_type="all"):
#         self.filter_type = filter_type


#     def __call__(self, results):

#         boxes = results['gt_bboxes_3d']
#         if len(boxes.tensor) <= 0:
#             return results
#         extrinsic = results['lidar2img']['extrinsic'][0]

#         boxes_cam = boxes.convert_to(Box3DMode.CAM, extrinsic)

#         if self.filter_type == "all":
#             mask = (boxes_cam.corners[..., 2] > 0).min(1)[0]
#         elif self.filter_type == "center":
#             mask = (boxes_cam.center[:, 2] > 0)
#         try:
#             results['gt_labels_3d'] = results['gt_labels_3d'][mask]
#         except:
#             return results
#         results['gt_bboxes_3d'].tensor = results['gt_bboxes_3d'].tensor[mask]
#         # results['gt_names'] = results['gt_names'][mask]
#         return results





@PIPELINES.register_module()
class ProjectLidar2Image(object):
    '''
        Project the point cloud to image using camera projection matrix

        check_flip: if flip augmentation is adopted on the image, flip the depth map (x, y, z)

    '''
    def __init__(self, check_flip=True):
        self.check_flip = check_flip


    def __call__(self, input_dict):
        points = input_dict['points']
        dense_depth = []
        for idx in range(len(input_dict['img'])):
            dense_depth_idx = np.zeros_like(input_dict['img'][idx])

            extrinsic = input_dict['lidar2cam'][idx]
            intrinsic = input_dict['cam2img'][idx]
            if isinstance(intrinsic, list):
                intrinsic = intrinsic[idx]
            points_numpy = points.tensor.numpy()
            points_numpy[:,-1] = 1
            points_cam = points_numpy @ extrinsic.T

            points_img = points_cam @ intrinsic.T

            points_img[:, 0] /= np.clip(points_img[:,2], a_min=1e-4, a_max=100000)
            points_img[:, 1] /= np.clip(points_img[:,2], a_min=1e-4, a_max=100000)
            img_shape = input_dict['img_shape']
            if isinstance(img_shape, list):
                img_shape = img_shape[0]
                mask = (points_cam[:,2] > 0) & \
                    (points_img[:,1] < img_shape[0]) & (points_img[:,1] > 0) & \
                    (points_img[:,0] < img_shape[1]) & (points_img[:,0] > 0)

            u = points_img[:,0].astype(np.long)[mask]
            v = points_img[:,1].astype(np.long)[mask]
            # depth = points_cam[mask,:3]

            dense_depth_idx[v, u] = points_cam[mask,:3]
            width = img_shape[1]
            if 'flip' in input_dict and input_dict['flip']:
                dense_depth_idx = self.flip_depth(dense_depth_idx, extrinsic, intrinsic, width)


            dense_depth_idx = np.ascontiguousarray(dense_depth_idx)
            dense_depth_idx = torch.tensor(dense_depth_idx)
            dense_depth.append(dense_depth_idx)

        input_dict['dense_depth'] = dense_depth
        seg_fields = input_dict.get("seg_fields", [])
        seg_fields.append("dense_depth")
        input_dict["seg_fields"] = seg_fields
        return input_dict

    def flip_depth(self, dense_depth, extrinsic, intrinsic, width):
        proj_matrix = intrinsic
        # do the flip operation
        # TODO replace it by torch.flip
        dense_depth = dense_depth[:,::-1]
        x = dense_depth[:,:,0]
        f = proj_matrix[0, 0]
        u = proj_matrix[0, 2]
        T_x = proj_matrix[0, 3]
        d = dense_depth[:,:,2]
        x = (d * width - 2 * u * d - 2 * T_x - f * x) / f

        dense_depth[:,:,0] = x
        return dense_depth





# Order: should after the bounding box
@PIPELINES.register_module()
class GenerateNocs(object):
    """
    1. Generate nocs based on lidar points
    2. based on the generated nocs, generate the partial foreground mask.

    """

    def __init__(self, check_flip=True, box_outside_range=0.01):
        self.check_flip = check_flip
        self.box_outside_range = box_outside_range

    def __call__(self, input_dict):

        object_coordinate = []

        dense_dimension = []
        dense_location = []
        dense_yaw = []

        foreground_mask = []


        valid_mask = []

        for idx in range(len(input_dict['img'])):
            object_coordinate_idx = np.zeros_like(input_dict['img'][idx])
            dense_dimension_idx = np.zeros_like(object_coordinate_idx)
            dense_location_idx = np.zeros_like(dense_dimension_idx)

            # valid_mask_idx = np.zeros_like(nocs_idx)
            valid_mask_idx = np.zeros([object_coordinate_idx.shape[0], object_coordinate_idx.shape[1], 1])

            foreground_mask_idx = np.zeros_like(valid_mask_idx)
            # foreground_mask_valid = np.zeros_lik
            dense_yaw_idx = np.zeros_like(valid_mask_idx)
            points = input_dict["dense_depth"][idx]
            extrinsic = input_dict['lidar2cam'][idx]
            intrinsic = input_dict['cam2img'][idx]
            cam_box = input_dict['gt_bboxes_3d'].convert_to(
                                    Box3DMode.CAM, rt_mat=extrinsic)
            valid_lidar_mask = points[..., -1] > 0
            # points = points.numpy()

            # for box in cam_box.tensor:
            #     # 1. project points to camera coordinate

            # for idx in
            indices = cam_box.tensor[:,2].sort(descending=True)[1]
            cam_box.tensor = cam_box.tensor[indices]
            img_shape = input_dict['img_shape']
            if isinstance(img_shape, list):
                img_shape = img_shape[0]
            cam_box_2d = projected_2d_box(cam_box,
                            rt_mat=torch.Tensor(intrinsic), img_shape=img_shape)

            box_numpy = cam_box.tensor.clone().numpy()
            box_numpy[:,:3] = cam_box.gravity_center.numpy()
            box_numpy[:,3:6] += self.box_outside_range

            inside_mask = points_in_rbbox(points[valid_lidar_mask].numpy(),box_numpy,
                                          z_axis=1, origin=(0.5, 0.5, 0.5))


            # first setup the labels based on 2d box (it is not precise).
            for jdx, (box_jdx, box2d_jdx) in enumerate(zip(cam_box.tensor, cam_box_2d)):

                x1, y1, x2, y2 = box2d_jdx.long().cpu().numpy()

                center = box_jdx[:3]
                dim = box_jdx[3:6]

                yaw = box_jdx[6:7]
                gravity_center = center.clone()
                gravity_center[1] -= dim[1]/2.

                dense_location_idx[y1:y2,x1:x2] = gravity_center
                dense_dimension_idx[y1:y2,x1:x2] = dim
                dense_yaw_idx[y1:y2,x1:x2] = yaw

                foreground_mask_idx[y1:y2,x1:x2] = -1


            for jdx, box_jdx in enumerate(cam_box.tensor):

                points_jdx = points[valid_lidar_mask][inside_mask[:,jdx]]

                center = box_jdx[:3]
                dim = box_jdx[3:6]
                yaw = box_jdx[6:7]
                gravity_center = center.clone()
                gravity_center[1] -= dim[1]/2.
                # fix it
                object_coordinate_jdx = points_jdx - gravity_center[:3].unsqueeze(0) # consider nocs later

                #nocs_idx = self.normalize_ocs(ocs_jdx, box)     # do the projection: 1.
                temp_mask = valid_lidar_mask.reshape(-1).clone()
                temp_mask[temp_mask.clone()] *= torch.tensor(inside_mask[:, jdx]).bool()
                object_coordinate_idx[temp_mask.reshape(valid_lidar_mask.shape)] = object_coordinate_jdx

                valid_mask_idx[temp_mask.reshape(valid_lidar_mask.shape)] = 1
                dense_location_idx[temp_mask.reshape(valid_lidar_mask.shape)] = gravity_center
                dense_dimension_idx[temp_mask.reshape(valid_lidar_mask.shape)] = dim
                dense_yaw_idx[temp_mask.reshape(valid_lidar_mask.shape)] = yaw
                foreground_mask_idx[temp_mask.reshape(valid_lidar_mask.shape)] = 1
                # dense_location_idx[]
                # nocs
            object_coordinate_idx = torch.tensor(object_coordinate_idx)
            # nocs = torch.tensor(nocs)
            object_coordinate.append(object_coordinate_idx)
            valid_mask.append(torch.tensor(valid_mask_idx))

            dense_location.append(torch.tensor(dense_location_idx))
            dense_dimension.append(torch.tensor(dense_dimension_idx))
            dense_yaw.append(torch.tensor(dense_yaw_idx))
            foreground_mask.append(torch.tensor(foreground_mask_idx))

        # pass
        input_dict["object_coordinate"] = object_coordinate
        input_dict["valid_coordinate_mask"] = valid_mask

        input_dict["dense_location"] = dense_location
        input_dict["dense_dimension"] = dense_dimension
        input_dict["dense_yaw"] = dense_yaw
        input_dict["foreground_mask"] = foreground_mask

        seg_fields = input_dict.get('seg_fields', [])
        seg_fields.extend(
            ['object_coordinate', "valid_coordinate_mask", "dense_location",
             "dense_dimension", "dense_yaw", "foreground_mask"])
        input_dict['seg_fields'] = seg_fields

        return input_dict



# TODO modify this part that can support multiview inputs
@PIPELINES.register_module()
class LoadDepthFromPoints(object):
    """
        Load pixel level depth by converting point cloud to image coordinate
        Args:

    """
    def __init__(self, to_float32=False, from_depth_completition=False):

        self.to_float32 = to_float32
        self.from_depth_completition=from_depth_completition

    def __call__(self, results):
        """Call function to load depth from point cloud
        """
        assert "points" in results

        if not self.from_depth_completition:
            extrinsic = results["lidar2cam"][0]
            intrinsic = results["cam2img"][0]
            # transformation_matrix =  @
            pts = results["points"]
            pts = pts.tensor
            pts[:,-1] = 1
            pts_camera = pts @ extrinsic.T
            pts_img = pts_camera @ intrinsic.T
            depth = np.zeros_like(results['img']).astype(np.float) # each channel represents x, y, z
            pts_img[:,:2] /= pts_img[:,-2:-1].clamp(min=1e-4)

            img_shape = results['img_shape']
            index_x = pts_img[:,0].floor()
            index_y = pts_img[:,1].floor()

            mask = (index_x >= 0) & (index_x < img_shape[1]) & (index_y >= 0) & (index_y < img_shape[0])
            # for x

            depth[index_y[mask].long(), index_x[mask].long(), 0] = pts_camera[mask, 0] #ignore x, during flip augmentation, x would shift to xxx
            # for y
            depth[index_y[mask].long(), index_x[mask].long(), 1] = pts_camera[mask, 1]
            # for z
            depth[index_y[mask].long(), index_x[mask].long(), 2] = pts_camera[mask, 2]
        else:
            depth_filename = results["filename"].replace("image_2", "depth_2")
            depth = Image.open(depth_filename)
            depth = np.array(depth) / 256.
            depth = depth[np.newaxis, :, :]
            w, h = depth.shape[1:]
            coord = np.concatenate([np.meshgrid(np.arange(h), np.arange(w))], axis=0)
            depth = np.concatenate([coord, depth, np.ones_like(depth)], axis=0)
            depth = np.linalg.inv(intrinsic.T) @ depth.reshape(4, 1)

            depth = depth.reshape((4, w, h))
            depth = depth[:3].transpose(1,2,0)
            # back project to 3D space
            # depth =

        results["depth"] = depth
        results["seg_fields"] = []
        results["seg_fields"].append("depth")

        return results





@PIPELINES.register_module()
class PseudoPointGenerator(object):
    '''
    Generate voxel point

    '''
    def __init__(self, anchor_generator, voxel_size):
        # use cpu due to multiprocess
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.voxel_range = anchor_generator['ranges'][0]

        self.voxel_size = voxel_size
        self.voxel_range = anchor_generator['ranges'][0]

        self.n_voxels = [
            round((self.voxel_range[3] - self.voxel_range[0]) /
                  self.voxel_size[0]),
            round((self.voxel_range[4] - self.voxel_range[1]) /
                  self.voxel_size[1]),
            round((self.voxel_range[5] - self.voxel_range[2]) /
                  self.voxel_size[2])
        ]
    def __call__(self, results):
        '''

        1. Generate voxel based on anchor generator
        2. convert the shape based on voxel shape

        '''

        results['pseudo_points'] = self.anchor_generator.grid_anchors(
                        [self.n_voxels[::-1]], device='cpu')[0][:, :3]

        voxel_shape = copy.deepcopy(self.n_voxels)
        voxel_shape = voxel_shape + [3]
        results['pseudo_points'] = results['pseudo_points'].reshape(voxel_shape).numpy()
        return results

@PIPELINES.register_module()

class PseudoPointToTensor(object):
    '''
    Generate voxel point

    '''
    def __init__(self):
        pass
    def __call__(self, results):
        results['pseudo_points'] = torch.Tensor(results['pseudo_points'].copy())
        return results




@PIPELINES.register_module()
class RandomFlipPseudoPoints(RandomFlip):
    '''
    Generate voxel point

    '''

    def __init__(self, voxel_range=[-25.0, -50.0, -2, 50.0, 50.0, 4],
                    **kwargs):

        super().__init__(**kwargs)
        # TODO consider when the coordinate is not symmetry.
        self.voxel_range = voxel_range


    def bbox_flip(self, bboxes, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        # do the 3d box flip


        # assert bboxes.shape[-1] % 4 == 0
        # flipped = bboxes.copy()
        bboxes.flip(direction)

        return bboxes

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['pseudo_points_flip'] = cur_dir is not None

        # do the pseudo points flip

        if 'flip_direction' not in results:
            results['pseudo_points_flip_direction'] = cur_dir
        if results['pseudo_points_flip']:
            # flip image
            if 'pseudo_points' in results:
                results['pseudo_points'] = mmcv.imflip(
                    results['pseudo_points'], direction=results['pseudo_points_flip_direction'])

            # only consider 3d boxes:
            for key in ['gt_bboxes_3d']:
                if key in results:
                    results[key] = self.bbox_flip(results[key],
                                                  results['pseudo_points_flip_direction'])

            # # flip bboxes
            # for key in results.get('bbox_fields', []):
            #     results[key] = self.bbox_flip(results[key],
            #                                   results['img_shape'],
            #                                   results['flip_direction'])
            # # flip masks
            # for key in results.get('mask_fields', []):
            #     results[key] = results[key].flip(results['flip_direction'])

            # # flip segs
            # for key in results.get('seg_fields', []):
            #     results[key] = mmcv.imflip(
            #         results[key], direction=results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'





# crop do the crop flip

@PIPELINES.register_module()
class RandomCropPseudoPoints(RandomCrop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """
            Function to randomly crop pseudo_points, 3D bounding boxes.
        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        assert "pseudo_points" in results
        pseudo_points = results["pseudo_points"]
        margin_h = max(pseudo_points.shape[0] - crop_size[0], 0)
        margin_w = max(pseudo_points.shape[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
        pseudo_points = pseudo_points[crop_y1:crop_y2, crop_x1:crop_x2, ...]

        results["pseudo_points"] = pseudo_points

        if "gt_bboxes_3d" in results:
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)

        else:
            return results


    def __cal__(self, results):
        point_shape = results['pseudo_points'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results





@PIPELINES.register_module()
class MultiViewRandomFlip3D(RandomFlip):
    '''
    !!! Current only flip the image and add the flip flag;
    do not convert the 3D bounding boxes and instrinsic;
    '''
    def __init__(self,
                 sync_2d=True,
                 **kwargs):
        super(MultiViewRandomFlip3D, self).__init__(**kwargs)
        self.sync_2d = sync_2d



    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            results['img'] = [mmcv.imflip(img, results['flip_direction']) for img in results['img']]
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            self.random_flip_data_3d(results, direction=results['flip_direction'])



        return results

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str, optional): Flip direction.
                Default: 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        # for semantic segmentation task, only points will be flipped.
        # if 'bbox3d_fields' not in input_dict:
        #     input_dict['points'].flip(direction)
        #     return
        # if len(input_dict['bbox3d_fields']) == 0:  # test mode
        #     input_dict['bbox3d_fields'].append('empty_box3d')
        #     input_dict['empty_box3d'] = input_dict['box_type_3d'](
        #         np.array([], dtype=np.float32))
        # assert len(input_dict['bbox3d_fields']) == 1
        # for key in input_dict['bbox3d_fields']:
        #     if 'points' in input_dict:
        #         input_dict['points'] = input_dict[key].flip(
        #             direction, points=input_dict['points'])
        #     else:
        #         input_dict[key].flip(direction)
        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['ori_shape'][1]
            input_dict['centers2d'][..., 0] = \
                w - input_dict['centers2d'][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]



@PIPELINES.register_module()
class CustomLoadAnnotations3D(LoadAnnotations3D):
    def __init__(self,
                with_instance_ids=False,
                **kwargs,):
        super().__init__(**kwargs)

        self.with_instance_ids = with_instance_ids

    def __call__(self, results):
        results = \
            super(CustomLoadAnnotations3D, self).__call__(results)

        if self.with_instance_ids:
            results = self._load_instance_ids(results)
        return results
    def _load_instance_ids(self, results):
        """Private function to load track id.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_instance_ids'] = results['ann_info']['gt_instance_ids']
        return results

    def __repr__(self):
        indent_str = '    '

        repr_str = super().__repr__()
        repr_str += f'{indent_str}with_instance_ids={self.with_instance_ids})'
        return repr_str


@PIPELINES.register_module()
class CustomObjectRangeFilter(ObjectRangeFilter):
    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # if ''
        if 'gt_instance_ids' in input_dict:
            gt_instance_ids = input_dict['gt_instance_ids']
            input_dict['gt_instance_ids'] = gt_instance_ids[mask.numpy().astype(np.bool)]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict



@PIPELINES.register_module()
class CustomMultiViewWrapper(object):
    '''Wrapper for processing multiview image.
    Args:


    '''
    def __init__(self,
                 transforms: dict,
                 override_aug_config: bool = True,
                 load_mono_anns: bool = False,
                 process_fields: list = [
                    'img_filename', 'cam2img', 'lidar2img'],
                 collected_keys: list = ['img',
                     'scale', 'scale_factor', 'crop', 'img_crop_offset',
                     'ori_shape', 'pad_shape', 'img_shape', 'pad_fixed_size',
                     'pad_size_divisor', 'flip', 'flip_direction', 'rotate',
                     # 'centers', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
                 ],
                 randomness_keys: list = [
                     'scale', 'crop_size', 'img_crop_offset',
                     'flip', 'flip_direction', 'photometric_param'
                 ],
                 preserved_keys: list = [
                    'lidar2img', 'cam2img', 'lidar2cam'],
                 mono_label_keys: list = [
                    'gt_bboxes', 'gt_labels', 'attr_labels',
                    'gt_bboxes_3d', 'gt_labels_3d', 'centers2d',
                    'depths']):
        self.load_mono_anns = load_mono_anns
        self.transforms = Compose(transforms)
        self.override_aug_config = override_aug_config
        self.collected_keys = collected_keys
        self.process_fields = process_fields
        self.randomness_keys = randomness_keys
        self.preserved_keys = preserved_keys
        self.mono_label_keys = mono_label_keys

    def __call__(self, input_dict):
        """Transform function to do the transform for multiview image.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: output dict after transformtaion
        """
        # store the augmentation related keys for each image.
        # save the origin key
        for key in self.preserved_keys:
            if key in input_dict:
                input_dict['ori_' + key] = copy.deepcopy(input_dict[key])

        for key in self.collected_keys:
            if key not in input_dict or \
                    not isinstance(input_dict[key], list):
                input_dict[key] = []

        if self.load_mono_anns:
            for key in self.mono_label_keys:
                key = "mono_" + key
                if key not in input_dict:
                    input_dict[key] = []
        prev_process_dict = {}
        for img_id in range(len(input_dict['img_filename'])):
            process_dict = dict(
                img_prefix=None,
                bbox_fields=[],
                bbox3d_fields=[])
            # override the process dict (e.g. scale in random scale,
            # crop_size in random crop, flip, flip_direction in
            # random flip)
            if img_id != 0 and self.override_aug_config:
                for key in self.randomness_keys:
                    if key in prev_process_dict:
                        process_dict[key] = prev_process_dict[key]
            for key in self.process_fields:
                if key in input_dict:
                    process_dict[key] = input_dict[key][img_id]
            if 'img_filename' in process_dict and 'img_info' not in process_dict:
                # convert the format to satisfy LoadImageFromFile
                process_dict['img_info'] = dict(
                    filename=process_dict['img_filename'])

            if self.load_mono_anns:
                process_dict['ann_info'] = \
                    input_dict['mono_ann_info'][img_id]

            process_dict = self.transforms(process_dict)
            # store the randomness variable in transformation.
            prev_process_dict = process_dict

            # store the related results to results_dict
            for key in self.process_fields:
                if key in process_dict:
                    input_dict[key][img_id] = process_dict[key]
            # update the keys
            for key in self.collected_keys:
                if key in process_dict:
                    if len(input_dict[key]) == img_id + 1:
                        input_dict[key][img_id] = process_dict[key]
                    else:
                        input_dict[key].append(process_dict[key])
            for key in self.mono_label_keys:
                if key in process_dict:
                    input_dict["mono_" + key].append(process_dict[key])
        for key in self.collected_keys:
            if len(input_dict[key]) == 0:
                input_dict.pop(key)

        return input_dict
