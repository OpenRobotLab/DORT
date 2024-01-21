import numpy as np
import copy
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import LoadMultiViewImageFromFiles
from mmdet.datasets.pipelines import RandomFlip, Resize, RandomCrop
from mmdet3d.datasets.pipelines.loading import LoadAnnotations
from det3d.core.bbox.util import projected_2d_box
from mmdet3d.datasets.pipelines import LoadAnnotations3D

def get_padded_shape(img_shape):
    img_shape = np.stack(img_shape, axis=0)
    img_shape_max = np.max(img_shape, axis=0)
    img_shape_min = np.min(img_shape, axis=0)
    assert img_shape_min[-1] == img_shape_max[-1]
    if np.all(img_shape_max == img_shape_min):
        # do not need to do the padding
        return None
    else:
        return img_shape_max[:2]

@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk'),
                 num_views=5,
                 num_ref_frames=-1,
                 test_mode=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.num_views = num_views
        # num_ref_frames is used for multi-sweep loading
        self.num_ref_frames = num_ref_frames
        # when test_mode=False, we randomly select previous frames
        # otherwise, select the earliest one
        self.test_mode = test_mode

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    raise NotImplementedError
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            for key in ['lidar2img', 'cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            # Transform lidar2img and lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2img', 'lidar2cam']:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename = results['img_filename']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [self.file_client.get(name) for name in filename]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        if 'cam2img' in results:
            results['ori_cam2img'] = copy.deepcopy(results['cam2img'])
        if 'lidar2img' in results:
            results['ori_lidar2img'] = copy.deepcopy(results['lidar2img'])
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'num_views={self.num_views}, '
        repr_str += f'num_ref_frames={self.num_ref_frames}, '
        repr_str += f'test_mode={self.test_mode})'
        return repr_str


@PIPELINES.register_module()
class CustomMultiViewImagePad(object):
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

        # should be the max shape of multi-view images
        results['img_shape'] = [img.shape for img in results['img']]

        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results['img']
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val)
                for img in results['img']
            ]
        results['img'] = padded_img

        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg_mask(self, results):
        # TODO: set a custom value for seg_mask padding
        # temporarily use pad_val=self.pad_val
        for key in results.get('seg_fields', []):
            padded_mask = [
                mmcv.impad(
                    mask,
                    shape=results['pad_shape'][0][:2],
                    pad_val=self.pad_val) for mask in results[key]
            ]
            results[key] = padded_mask

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg_mask(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class CustomMultiViewImageNormalize(object):
    """Normalize the image.

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
        results['img'] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results['img']
        ]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += \
            f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class CustomMultiViewImagePhotoMetricDistortion(object):
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
                'PhotoMetricDistortion needs the input image of dtype '\
                "np.float32, please set 'to_float32=True' in "\
                "'LoadImageFromFile' pipeline"
            # random brightness
            if np.random.randint(2):
                delta = np.random.uniform(-self.brightness_delta,
                                          self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                              self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(self.saturation_lower,
                                                 self.saturation_upper)

            # random hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue_delta,
                                                 self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                              self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
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
class CustomResize3D(Resize):
    """Resize 3D labels."""

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        super(CustomResize3D, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=keep_ratio,
            bbox_clip_border=bbox_clip_border,
            backend=backend,
            override=override)

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``ori_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """
        # ori_scale = results['img'].shape[:2]
        # consider the ori_scale can be specified by self.img_scale
        if self.img_scale is not None:
            ori_scale = self.img_scale[0]
        else:
            ori_scale = results['img'].shape[:2]
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                ori_scale, self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_3d(self, results):
        """Resize centers2d and modify camera intrinisc with
        ``results['scale']``."""
        if 'centers2d' in results:
            results['centers2d'] *= results['scale_factor'][:2]
        # resize image equals to change focal length and
        # camera intrinsic
        results['cam2img'][0] *= results['scale_factor'][0].repeat(
            len(results['cam2img'][0]))
        results['cam2img'][1] *= results['scale_factor'][1].repeat(
            len(results['cam2img'][1]))

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        super(Resize3D, self).__call__(results)
        self._resize_3d(results)

        return results



@PIPELINES.register_module()
class CustomResize3DPGD(Resize):
    """Resize 3D labels."""

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        super(CustomResize3DPGD, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=keep_ratio,
            bbox_clip_border=bbox_clip_border,
            backend=backend,
            override=override)

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``ori_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """
        # ori_scale = results['img'].shape[:2]
        # consider the ori_scale can be specified by self.img_scale
        if self.img_scale is not None:
            ori_scale = self.img_scale[0]
        else:
            ori_scale = results['img'].shape[:2]
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                ori_scale, self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_3d(self, results):
        """Resize centers2d and modify camera intrinisc with
        ``results['scale']``."""
        if 'centers2d' in results:
            results['centers2d'] *= results['scale_factor'][:2]
        # resize image equals to change focal length and
        # camera intrinsic
        results['cam2img'] = copy.deepcopy(results['cam2img'])
        results['cam2img'][0] *= results['scale_factor'][0].repeat(
            len(results['cam2img'][0]))
        results['cam2img'][1] *= results['scale_factor'][1].repeat(
            len(results['cam2img'][1]))

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \zx
                'keep_ratio' keys are added into result dict.
        """
        super(CustomResize3DPGD, self).__call__(results)
        self._resize_3d(results)

        return results


@PIPELINES.register_module()
class CustomRandomFlip3DPGD(RandomFlip):
    """Flip the points & bbox.
    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 flip_points = True,
                 **kwargs):
        super(CustomRandomFlip3DPGD, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        self.flip_points = flip_points
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

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
        if 'bbox3d_fields' not in input_dict:
            input_dict['points'].flip(direction)
            return
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

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        # flip 2D image and its annotations
        super(CustomRandomFlip3DPGD, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical
        if self.flip_points:
            if 'transformation_3d_flow' not in input_dict:
                input_dict['transformation_3d_flow'] = []

            if input_dict['pcd_horizontal_flip']:
                self.random_flip_data_3d(input_dict, 'horizontal')
                input_dict['transformation_3d_flow'].extend(['HF'])
            if input_dict['pcd_vertical_flip']:
                self.random_flip_data_3d(input_dict, 'vertical')
                input_dict['transformation_3d_flow'].extend(['VF'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str


@PIPELINES.register_module()
class CustomMultiViewImageResize3D(CustomResize3D):
    """Random scale the image with modifying camera intrinsic. This function is
    still unstable, required further testing.

    Args:
        img_scale: resolution for the output image
        keep_ratio: keep the ratio between w and h
        resize_depth: consider simltaneously resize depth (for training aug)
        ratio_range:
        override: do the resize more than one time.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        super(CustomMultiViewImageResize3D, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=keep_ratio,
            bbox_clip_border=bbox_clip_border,
            backend=backend,
            override=override)
        # temporarily do not need these params
        # because resize_depth=False & resize_intrinsic=True
        # is an optimal setting surveyed before
        # self.resize_depth = resize_depth
        # self.rescale_intrinsic = rescale_intrinsic

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``ori_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """
        # NOTE: we only support a single scale for multi-view for now
        # TODO: support different scales for multi-view images
        # ori_scale = results['img'].shape[:2]
        # consider the ori_scale can be specified by self.img_scale
        if self.img_scale is not None:
            ori_scale = self.img_scale[0]
        else:
            ori_scale = results['img'][0].shape[:2]
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                ori_scale, self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_3d(self, results):
        """Resize centers2d and modify camera intrinisc with
        ``results['scale']``."""
        if 'centers2d' in results:
            results['centers2d'] *= results['scale_factor'][:2]
        # resize image equals to change focal length and
        # camera intrinsic
        if 'cam2img' in results:
            for idx, cam2img in enumerate(results['cam2img']):
                cam2img[0] *= results['scale_factor'][0].repeat(
                    len(cam2img[0]))
                cam2img[1] *= results['scale_factor'][1].repeat(
                    len(cam2img[0]))
                results['cam2img'][idx] = cam2img
                results['lidar2img'][idx] = cam2img @ results['lidar2cam'][idx]

    def _resize_img(self, results):
        """Resize images with list of inputs ``results['scale']``."""

        # other with type of tensor
        for idx, img in enumerate(results['img']):
            if self.keep_ratio:
                resized_img, scale_factor = mmcv.imrescale(
                    img,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = resized_img.shape[:2]
                h, w = img.shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                resized_img, w_scale, h_scale = mmcv.imresize(
                    img,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results['img'][idx] = resized_img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)

        results['img_shape'] = results['img'][0].shape
        results['img_resized_shape'] = [img.shape for img in results['img']]
        # in case that there is no padding
        results['pad_shape'] = results['img'][0].shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            for idx, seg in enumerate(results[key]):
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        seg,
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    gt_seg = mmcv.imresize(
                        seg,
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                results[key][idx] = gt_seg

    def __call__(self, results):
        # self.drop_ratio = drop_ratio
        # super(RandomScaleImage3D, self).__call__(input_dict)
        # Assume the multiview image is in the same shape
        # currently img is with the shape of [N, C, H, W, M]

        # NOTE: remove the check scale_factor part due to multi-view imgs
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
        self._resize_3d(results)
        return results


@PIPELINES.register_module()
class CustomRandomCrop3D(RandomCrop):
    """3D version of RandomCrop.

    RamdomCrop3D needs some customized settings and modifications for camera
    intrinsic matrix.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 recompute_bbox=False,
                 bbox_clip_border=True,
                 rel_offset_h=(0., 1.),
                 rel_offset_w=(0., 1.)):
        super().__init__(
            crop_size=crop_size,
            crop_type=crop_type,
            allow_negative_crop=allow_negative_crop,
            recompute_bbox=recompute_bbox,
            bbox_clip_border=bbox_clip_border)
        # rel_offset specifies the relative offset range of cropping origin
        # [0., 1.] means starting from 0*margin to 1*margin + 1
        self.rel_offset_h = rel_offset_h
        self.rel_offset_w = rel_offset_w

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

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
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            # TOCHECK: a little different from LIGA implementation
            offset_h = np.random.randint(self.rel_offset_h[0] * margin_h,
                                         self.rel_offset_h[1] * margin_h + 1)
            offset_w = np.random.randint(self.rel_offset_w[0] * margin_w,
                                         self.rel_offset_w[1] * margin_w + 1)
            crop_h = min(crop_size[0], img.shape[0])
            crop_w = min(crop_size[1], img.shape[1])
            crop_y1, crop_y2 = offset_h, offset_h + crop_h
            crop_x1, crop_x2 = offset_w, offset_w + crop_w

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # manipulate camera intrinsic matrix
        # needs to apply offset to K instead of P2 (on KITTI)
        K = results['cam2img'][:3, :3].copy()
        inv_K = np.linalg.inv(K)
        T = np.matmul(inv_K, results['cam2img'][:3])
        K[0, 2] -= crop_x1
        K[1, 2] -= crop_y1
        offset_cam2img = np.matmul(K, T)
        results['cam2img'][:offset_cam2img.shape[0], :offset_cam2img.
                           shape[1]] = offset_cam2img
        results['crop_offset'] = [crop_x1, crop_y1]

        return results

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border}), '
        repr_str += f'rel_offset_h={self.rel_offset_h}), '
        repr_str += f'rel_offset_w={self.rel_offset_w})'
        return repr_str





@PIPELINES.register_module()
class CustomMultiViewImageCrop3D(CustomRandomCrop3D):
    """Random crop the image with recording the crop index. This function is
    still unstable, required futher testing.

    Args:


    """
    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 recompute_bbox=False,
                 bbox_clip_border=True,
                 rel_offset_h=(0., 1.),
                 rel_offset_w=(0., 1.)):
        super().__init__(
            crop_size=crop_size,
            crop_type=crop_type,
            allow_negative_crop=allow_negative_crop,
            recompute_bbox=recompute_bbox,
            bbox_clip_border=bbox_clip_border)
        # rel_offset specifies the relative offset range of cropping origin
        # [0., 1.] means starting from 0*margin to 1*margin + 1
        self.rel_offset_h = rel_offset_h
        self.rel_offset_w = rel_offset_w


    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        !!! Currently I did not modify the cam instrinsic
            because imVoxelNet use the origin cam instrinsic;
            In the future,  if modify instrinsic, one may consider the conflict with flip augmentation.

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
        for idx, img in enumerate(results['img']):
            if idx == 0:
                margin_h = max(img.shape[0] - crop_size[0], 0)
                margin_w = max(img.shape[1] - crop_size[1], 0)
                # TOCHECK: a little different from LIGA implementation
                offset_h = np.random.randint(self.rel_offset_h[0] * margin_h,
                                            self.rel_offset_h[1] * margin_h + 1)
                offset_w = np.random.randint(self.rel_offset_w[0] * margin_w,
                                            self.rel_offset_w[1] * margin_w + 1)
                crop_h = min(crop_size[0], img.shape[0])
                crop_w = min(crop_size[1], img.shape[1])
                crop_y1, crop_y2 = offset_h, offset_h + crop_h
                crop_x1, crop_x2 = offset_w, offset_w + crop_w
            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape

            results['img'][idx] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # manipulate camera intrinsic matrix
        # needs to apply offset to K instead of P2 (on KITTI)
        # K = results['cam2img'][:3, :3].copy()
        # inv_K = np.linalg.inv(K)
        # T = np.matmul(inv_K, results['cam2img'][:3])
        # K[0, 2] -= crop_x1
        # K[1, 2] -= crop_y1
        # offset_cam2img = np.matmul(K, T)
        # results['cam2img'][:offset_cam2img.shape[0], :offset_cam2img.
                        #    shape[1]] = offset_cam2img
        results['img_crop_offset'] = [crop_x1, crop_y1]

        return results

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        if isinstance(results['img'], list):
            image_size = results['img'][0].shape[:2]
        else:
            image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border}), '
        repr_str += f'rel_offset_h={self.rel_offset_h}), '
        repr_str += f'rel_offset_w={self.rel_offset_w})'
        return repr_str


@PIPELINES.register_module()
class CustomMultiViewRandomFlip3D(RandomFlip):
    '''
    !!! Current only flip the image and add the flip flag;
    do not convert the 3D bounding boxes and instrinsic;
    '''
    def __init__(self,
                 sync_2d=True,
                 **kwargs):
        super(CustomMultiViewRandomFlip3D, self).__init__(**kwargs)
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
class LoadMultipleMonoAnnotations3D(LoadAnnotations3D):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """
    def __init__(self,
                 sync_2d=True,
                 sync_3d=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.sync_3d = sync_3d
        self.sync_2d = sync_2d

    def __call__(self, results):
        """
        Temp
        """
        mono_results = []
        for idx in range(len(results['mono_ann_info'])):
            sub_results = dict(
                img_prefix=None,
                bbox_fields=[],
                bbox3d_fields=[])  
            # cam2img = results['']
            sub_results['ann_info'] = results['mono_ann_info'][idx]
            sub_results = super().__call__(sub_results)

            sub_results['scale_factor'] = results['scale_factor'][idx]
            sub_results['img_shape'] = results['img_shape'][idx]
            sub_results['ori_img_shape'] = results['ori_img_shape'][idx]

            sub_results['cam2img'] = results['cam2img'][idx]
            self._resize_3d(sub_results)
            self._resize_bboxes(sub_results)
            flip = results['flip'][idx]
            if flip == 1:
                self.random_flip_data_3d(sub_results)
            mono_results.append(sub_results)

        

        if self.with_bbox:
            results['mono_gt_bboxes'] = [
                i['gt_bboxes'] for i in mono_results]
            results['mono_gt_bboxes_ignore'] = [
                i['gt_bboxes_ignore'] for i in mono_results]
        if self.with_label:
            results['mono_gt_labels'] = [
                i['gt_labels'] for i in mono_results]
        if self.with_attr_label:
            results['mono_attr_labels'] = [
                i['attr_labels'] for i in mono_results]
        if self.with_label_3d:
            results['mono_gt_labels_3d'] = [
                i['gt_labels_3d'] for i in mono_results]
            results['mono_gt_bboxes_3d'] = [
                i['gt_bboxes_3d'] for i in mono_results]
        if self.with_bbox_depth:
            results['mono_depths'] = [
                i['depths'] for i in mono_results]
            results['mono_centers2d'] = [
                i['centers2d'] for i in mono_results]    
        if 'cam2img' in mono_results[0]:
            results['mono_cam2img'] = [
                i['cam2img'] for i in mono_results]
        return results


    def _resize_3d(self, results):


        """Resize centers2d and modify camera intrinisc with
        ``results['scale']``."""
        if 'centers2d' in results:
            results['centers2d'] *= results['scale_factor'][:2]
        # resize image equals to change focal length and
        # camera intrinsic
        results['cam2img'] = copy.deepcopy(results['cam2img'])
        results['cam2img'][0] *= results['scale_factor'][0]
        results['cam2img'][1] *= results['scale_factor'][1]

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key][:,0::2] * results['scale_factor'][0]
            bboxes = results[key][:,1::2] * results['scale_factor'][1]

            
            img_shape = results['img_shape']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    

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
        if 'bbox3d_fields' not in input_dict:
            input_dict['points'].flip(direction)
            return
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        if self.sync_3d:
            for key in input_dict['bbox3d_fields']:
                if 'points' in input_dict:
                    input_dict['points'] = input_dict[key].flip(
                        direction, points=input_dict['points'])
                else:
                    input_dict[key].flip(direction)
        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['img_shape'][1]
            input_dict['centers2d'][..., 0] = \
                w - input_dict['centers2d'][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]



    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped