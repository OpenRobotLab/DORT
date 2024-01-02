# from .dataset_wrappers import MultiViewMixin
import numpy as np
import warnings

from mmdet.datasets import DATASETS
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets.custom_3d import Custom3DDataset
from .nuscenes_dataset import CustomNuScenesDataset
from mmdet3d.datasets import NuScenesMonoDataset
from mmdet3d.core.bbox import get_box_type
import mmcv


@DATASETS.register_module()
class CustomNuScenesFrameDataset(CustomNuScenesDataset, NuScenesMonoDataset):
    r"""NuScenes Dataset.

    Used for training.
    This dataset read both the whole frame bbox and the mono bbox.
    ref_img_sampler for tracking dataset, default is None,
    an example of ref_img_sampler:
        ref_img_sampler=dict(
            frame_range=[-3, 3],
            stride=1,
            num_ref_imgs=1)c
    """
    NUM_IMGS = 6
    def __init__(self,
                 ann_file=None,
                 mono_ann_file=None,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 ref_img_sampler=None,
                 max_repeat_time = 100,
                 post_pipeline = None,
                 file_client_args=dict(backend='disk'),
                 **kwargs):
        self.ref_img_sampler = ref_img_sampler
        self.max_repeat_time = max_repeat_time
        if post_pipeline is not None:
            self.post_pipeline = Compose(post_pipeline)


        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.bbox_code_size = 9
        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        # load annotations
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(open(local_path, 'rb'))
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)

        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        if self.eval_version == "detection_cvpr_2019":
            from nuscenes.eval.detection.config import config_factory
            self.eval_detection_configs = config_factory(self.eval_version)
        elif self.eval_version == "tracking_nips_2019":
            from nuscenes.eval.common.config import config_factory
            self.eval_detection_configs = config_factory(self.eval_version)

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,)
            
        if mono_ann_file is not None:
            self.mono_data_infos = self.load_mono_annotations(mono_ann_file)

    def load_mono_annotations(self, ann_file):
        return NuScenesMonoDataset.load_annotations(self, ann_file)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        # data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data['infos'][::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        # check how to format the output;
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        # for the tracking dataset
        for key in ['scene_token', 'prev', 'is_first_frame']:
            if key in info:
                input_dict[key] = info[key]

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam2imgs = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam2imgs.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam2img=cam2imgs,
                    lidar2cam=lidar2cam_rts,
                ))
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            mono_annos = []
            for i in range(self.NUM_IMGS):
                mono_annos.append(
                    self.get_mono_ann_info(index*self.NUM_IMGS + i))
            input_dict['mono_ann_info'] = mono_annos
        return input_dict

    def get_mono_ann_info(self, index):
        img_id = self.mono_data_infos[index]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.mono_data_infos[index], ann_info)