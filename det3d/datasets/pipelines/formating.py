from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import Collect3D, DefaultFormatBundle3D
import numpy as np
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D

@PIPELINES.register_module()
class CustomCollect3D(Collect3D):
    """
        Add lidar2cam, intrinsics to the meta keys
    
    """
    
    def __init__(
        self,
        keys,
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'lidar2cam', 'cam2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                   'pts_filename', 'transformation_3d_flow', 'trans_mat',
                   'affine_aug', 'ori_lidar2img', 'img_crop_offset',
                   'img_resized_shape', 'num_ref_frames', 'num_views', 
                   'is_first_frame', 'scene_token', 
                   'ego2global', 'transformation_3d_flow',
                   'lidar2ego_translation', 'lidar2ego_rotation',
                   'ego2global_translation', 'ego2global_rotation',
                   'ori_lidar2img', 'ori_cam2img', 'ori_lidar2cam',
                   'mono_cam2img')):
        
        super().__init__(keys, meta_keys)
@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    def __init__(self,
                 process_img=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.process_img = process_img
    
    def __call__(self, results):
        if self.process_img is False:
            img = results.pop("img")
            results = super().__call__(results)
            results['img'] = img
        else:
            results = super().__call__(results)
        return results


@PIPELINES.register_module()
class SeqFormating(object):
    def __init__(self, ref_prefix='ref'):
        self.ref_prefix = ref_prefix

    def __call__(self, results):
        if results is None:
            return None
        # currently we only support one ref images.
        assert len(results) == 2

        data = {}
        data.update(results[0])

        for k, v in results[1].items():
            data[f'{self.ref_prefix}_{k}'] = v
        return data


@PIPELINES.register_module()
class CustomMatchInstances(object):
    """Matching objects on a pair of images.

    Args:
        skip_nomatch (bool, optional): Whether skip the pair of image
        during training when there are no matched objects. Default
        to True.
    """
    def __init__(self, skip_nomatch=True):
        self.skip_nomatch = skip_nomatch
    def _match_gts(self, instance_ids, ref_instance_ids):
        """Matching objects according to ground truth `instance_ids`.

        Args:
            instance_ids (ndarray): of shape (N1, ).
            ref_instance_ids (ndarray): of shape (N2, ).

        Returns:
            tuple: Matching results which contain the indices of the
            matched target.
        """
        ins_ids = list(instance_ids)
        ref_ins_ids = list(ref_instance_ids)
        match_indices = np.array([
            ref_ins_ids.index(i) if (i in ref_ins_ids and i != '') else -1
            for i in ins_ids
        ])
        ref_match_indices = np.array([
            ins_ids.index(i) if (i in ins_ids and i != '') else -1
            for i in ref_ins_ids
        ])
        return match_indices, ref_match_indices

    def __call__(self, results):
        if len(results) != 2:
            raise NotImplementedError('Only support match 2 images now.')

        match_indices, ref_match_indices = self._match_gts(
            results[0]['gt_instance_ids'], results[1]['gt_instance_ids'])
        nomatch = (match_indices == -1).all()
        if self.skip_nomatch and nomatch:
            return None
        else:
            results[0]['gt_match_indices'] = match_indices.copy()
            results[1]['gt_match_indices'] = ref_match_indices.copy()
        results[0].pop("gt_instance_ids")
        results[1].pop("gt_instance_ids")
        return results
