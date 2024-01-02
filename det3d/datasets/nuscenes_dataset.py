import os.path as osp
from pathlib import Path
from tkinter.messagebox import NO

# from .dataset_wrappers import MultiViewMixin
import mmcv
import numpy as np
import pyquaternion
import torch
from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
from mmdet3d.core.bbox import CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.core.bbox.structures import Box3DMode
from mmdet3d.core.visualizer import show_multi_modality_result
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.nuscenes_dataset import lidar_nusc_box_to_global
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Normalize
from nuscenes.utils.data_classes import Box as NuScenesBox
from tqdm import tqdm
import random
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets.custom_3d import Custom3DDataset


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    ref_img_sampler for tracking dataset, default is None,
    an example of ref_img_sampler:
        ref_img_sampler=dict(
            frame_range=[-3, 3],
            stride=1,
            num_ref_imgs=1)c
    """
    def __init__(self,
                 ann_file=None,
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
                 load_interval_shuffle=1,
                 ref_img_sampler=None,
                 max_repeat_time = 100,
                 post_pipeline = None,
                 **kwargs):
        self.load_interval_shuffle = float(load_interval_shuffle)
        self.ref_img_sampler = ref_img_sampler
        self.max_repeat_time = max_repeat_time
        if post_pipeline is not None:
            self.post_pipeline = Compose(post_pipeline)


        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        Custom3DDataset.__init__(
            self,
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        print(data_root)

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
                use_external=False,
            )


    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        if self.load_interval_shuffle > 1:
            selected_size = int(
                len(self.data_infos) / self.load_interval_shuffle)
            self.data_infos = np.random.choice(self.data_infos,
                                               selected_size,
                                               replace=True)

        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def ref_img_sampling(self, index, frame_range, stride=1, num_ref_imgs=1):
        assert stride==1
        index_list = [index]
        ref_pool = list(range(frame_range[0], frame_range[1]))
        ref_pool.remove(0)
        repeat_time = 1
        while len(index_list) < num_ref_imgs+1:
            if repeat_time > self.max_repeat_time:
                return None
            sample_index = random.sample(ref_pool, 1)[0] + index
            if sample_index in index_list:
                continue
            if sample_index >= len(self.data_infos):
                continue
            
            # check if scene token is the same
            if self.data_infos[index]['scene_token'] != \
                self.data_infos[sample_index]['scene_token']:
                continue
            index_list.append(sample_index)
        return index_list

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        if self.ref_img_sampler is None:
            results = super(CustomNuScenesDataset, self).prepare_train_data(index)
            return results
        else:
            # info = self.data_infos[index]
            index_list = self.ref_img_sampling(index, **self.ref_img_sampler)
            if index_list == None:
                return None
            results = [
                super(NuScenesDataset, self).prepare_train_data(index) for index in index_list]
            results = self.post_pipeline(results)
            return results

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        if self.ref_img_sampler is None:
            return super(CustomNuScenesDataset, self).prepare_test_data(index)
        else:
            index_list = self.ref_img_samping(index, **self.ref_img_sampler)
            if index_list == None:
                return None
            results = [
                super(CustomNuScenesDataset, self).prepare_test_data(index) for index in index_list]
            return self.post_pipeline(results)


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

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d,
                                            box_dim=gt_bboxes_3d.shape[-1],
                                            origin=(0.5, 0.5, 0.5)).convert_to(
                                                self.box_mode_3d)

        anns_results = dict(gt_bboxes_3d=gt_bboxes_3d,
                            gt_labels_3d=gt_labels_3d,
                            gt_names=gt_names_3d)

        if 'instance_token' in info:
            instance_token =info['instance_token'][mask]
            anns_results['gt_instance_ids'] = instance_token

        return anns_results

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        # modify from the partent class (NuScenesDataset)
        # to support the evaluation on the train subsetw

        if self.eval_version == "detection_cvpr_2019":
            from nuscenes import NuScenes
            from nuscenes.eval.detection.evaluate import NuScenesEval
        elif self.eval_version == "tracking_nips_2019":
            from nuscenes import NuScenes
            from nuscenes.eval.tracking.evaluate import TrackingEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version,
                        dataroot=self.data_root,
                        verbose=False)
        # eval_set_map = {
        #     'v1.0-mini': 'mini_val',
        #     'v1.0-trainval': 'val',
        # }
        if self.version == 'v1.0-mini':
            if 'train' in self.ann_file:
                eval_set = 'mini_train'
            else:
                eval_set = 'mini_val'
        elif self.version == 'v1.0-trainval':
            if 'train' in self.ann_file:
                eval_set = 'train'
            else:
                eval_set = 'val'
        if self.eval_version == "detection_cvpr_2019":
            nusc_eval = NuScenesEval(nusc,
                                    config=self.eval_detection_configs,
                                    result_path=result_path,
                                    eval_set=eval_set,
                                    output_dir=output_dir,
                                        verbose=False)
            nusc_eval.main(render_curves=False)

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
            detail = dict()
            metric_prefix = f'{result_name}_NuScenes'
            for name in self.CLASSES:
                for k, v in metrics['label_aps'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
                for k, v in metrics['label_tp_errors'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
                for k, v in metrics['tp_errors'].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}'.format(metric_prefix,
                                        self.ErrNameMapping[k])] = val

            detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
            detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
            return detail
        elif self.eval_version == "tracking_nips_2019":
            nusc_eval = TrackingEval(
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set,
                output_dir=output_dir,
                nusc_dataroot=self.data_root,
                nusc_version=self.version,
                verbose=True)
            metrics_summary = nusc_eval.main()
            metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
            detail=dict()
            metric_prefix = f'{result_name}_NuScenes'
            # check how to summary the metrics

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            if 'ids' in det:
                ids = det['ids']
            else:
                ids = None
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                if ids is not None:
                    nusc_anno['tracking_id'] = str(ids[i].item())
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path



@DATASETS.register_module()
class NuScenesSingleViewDataset(NuScenesDataset):
    """Based on NuScenes dataset, implement the monocular version for monocular
    detector."""
    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 pre_filter_empty_gt=False,
                 eval_version='detection_cvpr_2019',
                 load_interval_shuffle=1,
                 use_valid_flag=False):

        self.load_interval_shuffle = float(load_interval_shuffle)
        self.n_views = 6
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            classes=classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            eval_version=eval_version,
            use_valid_flag=use_valid_flag,
        )
        '''
        self.empty_mask = [False for idx in range(len(self))]

        if self.test_mode is False and pre_filter_empty_gt is True:
            self.empty_mask = self.filter_empty_gt_data()
        '''
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        if self.load_interval_shuffle > 1:
            selected_size = int(len(data_infos) / self.load_interval_shuffle)
            data_infos = np.random.choice(data_infos,
                                          selected_size,
                                          replace=True)

        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def filter_empty_gt_data(self):
        # new_data_infos = []
        # new_ann_infos = []
        empty_mask = []
        empty_num = 0
        # Adopt pickle processing
        pkl_name = self.ann_file.replace('.pkl', '') + '-gt_mask.pkl'
        if osp.exists(pkl_name):
            empty_mask, empty_num = mmcv.load(pkl_name)
        else:
            for idx in tqdm(range(len(self))):
                input_dict = self.get_data_info(idx)
                ann_info = input_dict['ann_info']

                extrinsic = input_dict['lidar2img']['extrinsic'][0]
                intrinsic = input_dict['lidar2img']['intrinsic']

                gt_bboxes = ann_info['gt_bboxes_3d'].convert_to(
                    Box3DMode.CAM, torch.tensor(extrinsic))

                projected_center = gt_bboxes.projected_gravity_center(
                    torch.tensor(intrinsic))

                mask = (projected_center[:, 0] > 0) & \
                       (projected_center[:, 1] > 0) & \
                       (projected_center[:, 0] < 1600) & \
                       (projected_center[:, 1] < 900) & \
                       (gt_bboxes.tensor[:, 2] > 0)
                if mask.sum() > 0:
                    empty_mask.append(False)
                else:
                    empty_num += 1
                    empty_mask.append(True)
            mmcv.dump([empty_mask, empty_num], pkl_name)
        print(f'num of example {len(self)}, empty num {empty_num}')
        return empty_mask

    def __len__(self):
        return len(self.data_infos) * self.n_views  # six cameras

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        # modify from the partent class (NuScenesDataset)
        # to support the evaluation on the train subsetw

        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version,
                        dataroot=self.data_root,
                        verbose=False)
        # eval_set_map = {
        #     'v1.0-mini': 'mini_val',
        #     'v1.0-trainval': 'val',
        # }
        if self.version == 'v1.0-mini':
            if 'train' in self.ann_file:
                eval_set = 'mini_train'
            else:
                eval_set = 'mini_val'
        elif self.version == 'v1.0-trainval':
            if 'train' in self.ann_file:
                eval_set = 'train'
            else:
                eval_set = 'val'
        print(eval_set)
        nusc_eval = NuScenesEval(nusc,
                                 config=self.eval_detection_configs,
                                 result_path=result_path,
                                 eval_set=eval_set,
                                 output_dir=output_dir,
                                 verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def get_data_info(self, index):
        # ori_index = index
        """if self.empty_mask[index] is True:

        return None
        """
        # pass

        cam_index = index % self.n_views
        index = index // self.n_views
        info = self.data_infos[index]

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        image_paths = []

        lidar2cam_rts = []
        lidar2img_rts = []
        camera_intrinsics = []
        # do the random select
        cam_type = list(info['cams'])[cam_index]
        cam_info = info['cams'][cam_type]
        lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
        lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_info['cam2img']
        intrinsic_pad = np.eye(4)
        intrinsic_pad[:3, :3] = intrinsic
        intrinsic = intrinsic_pad
        # viewpad = np.eye(4)
        # extrinsic = copy.deepcopy(lidar2cam_rt)
        # viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

        lidar2cam_rts.append(lidar2cam_rt.T.astype(np.float32))
        lidar2img_rt = (intrinsic @ lidar2cam_rt.T.astype(np.float32))
        lidar2img_rts.append(lidar2img_rt)
        camera_intrinsics.append(intrinsic.astype(np.float32))

        # intrinsics.append(intrinsic
        image_paths.append(cam_info['data_path'])
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                lidar2cam=lidar2cam_rts,
                cam2img=camera_intrinsics,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if len(annos['gt_bboxes_3d'].tensor) > 0:
                lidar2ego_translation = info['lidar2ego_translation']
                lidar2ego_rotation = info['lidar2ego_rotation']
                yaw = self.get_cam_yaw(annos['gt_bboxes_3d'].tensor,
                                       lidar2ego_translation,
                                       lidar2ego_rotation, cam_info)
                input_dict['ann_info']['gt_bboxes_3d'].tensor[:, 6] = yaw

        input_dict['img_prefix'] = [None]
        input_dict['img_info'] = [
            dict(filename=x) for x in input_dict['img_filename']
        ]

        if 'ann_info' in input_dict:
            # TODO filter data?

            # remove gt velocity
            gt_bboxes_3d = input_dict['ann_info']['gt_bboxes_3d'].tensor
            gt_bboxes_3d = gt_bboxes_3d[:, :-2]
            gt_bboxes_3d = self.box_type_3d(gt_bboxes_3d)

            gt_labels_3d = input_dict['ann_info']['gt_labels_3d'].copy()
            gt_names = input_dict['ann_info']['gt_names']

            input_dict['ann_info'] = dict(gt_bboxes_3d=gt_bboxes_3d,
                                          gt_names=gt_names,
                                          gt_labels_3d=gt_labels_3d)

        return input_dict

    def evaluate(self, results, *args, **kwargs):

        result_dict = super().evaluate(results, *args, **kwargs)
        return result_dict

    def get_attr_name(self, attr_idx, label_name):
        """Get attribute from predicted index.

        This is a workaround to predict attribute when the predicted velocity
        is not reliable. We map the predicted attribute index to the one
        in the attribute set. If it is consistent with the category, we will
        keep it. Otherwise, we will use the default attribute.

        Args:
            attr_idx (int): Attribute index.
            label_name (str): Predicted category name.

        Returns:
            str: Predicted attribute name.
        """
        # TODO: Simplify the variable name
        AttrMapping_rev2 = [
            'cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving',
            'pedestrian.standing', 'pedestrian.sitting_lying_down',
            'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None'
        ]
        if label_name == 'car' or label_name == 'bus' \
            or label_name == 'truck' or label_name == 'trailer' \
                or label_name == 'construction_vehicle':
            if AttrMapping_rev2[attr_idx] == 'vehicle.moving' or \
                AttrMapping_rev2[attr_idx] == 'vehicle.parked' or \
                    AttrMapping_rev2[attr_idx] == 'vehicle.stopped':
                return AttrMapping_rev2[attr_idx]
            else:
                return NuScenesSingleViewDataset.DefaultAttribute[label_name]
        elif label_name == 'pedestrian':
            if AttrMapping_rev2[attr_idx] == 'pedestrian.moving' or \
                AttrMapping_rev2[attr_idx] == 'pedestrian.standing' or \
                    AttrMapping_rev2[attr_idx] == \
                    'pedestrian.sitting_lying_down':
                return AttrMapping_rev2[attr_idx]
            else:
                return NuScenesSingleViewDataset.DefaultAttribute[label_name]
        elif label_name == 'bicycle' or label_name == 'motorcycle':
            if AttrMapping_rev2[attr_idx] == 'cycle.with_rider' or \
                    AttrMapping_rev2[attr_idx] == 'cycle.without_rider':
                return AttrMapping_rev2[attr_idx]
            else:
                return NuScenesSingleViewDataset.DefaultAttribute[label_name]
        else:
            return NuScenesSingleViewDataset.DefaultAttribute[label_name]

    def show(self, results, out_dir):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
        """

        for i, result in enumerate(results):
            example = self.prepare_test_data(i)
            img_metas = example['img_metas']._data
            img = example['img']._data.numpy()
            # remove the normalize operation
            img = img[0]
            normalize_transform = None
            for transform_candidate in self.pipeline.transforms[
                    0].transforms.transforms:
                if isinstance(transform_candidate, Normalize):
                    normalize_transform = transform_candidate
            if normalize_transform is not None:
                mean = normalize_transform.mean
                std = normalize_transform.std
                to_rgb = normalize_transform.to_rgb
                img = img.transpose(1, 2, 0)

                img = mmcv.imdenormalize(img, mean, std,
                                         to_bgr=to_rgb).astype(np.uint8)
                img = np.ascontiguousarray(img)
            else:
                img = img.transpose(1, 2, 0)

            filename = Path(img_metas['filename']).name
            inds = result['scores_3d'] > 0.01
            pred_bboxes = result['boxes_3d'][inds]
            # pred_bboxes = result.convert_to
            gt_bboxes = self.get_ann_info(i // 6)['gt_bboxes_3d']
            if len(pred_bboxes) == 0:
                pred_bboxes = None
            else:
                pred_bboxes = pred_bboxes.convert_to(
                    Box3DMode.LIDAR,
                    rt_mat=torch.inverse(img_metas['lidar2img']))

            # check when the tensor is zero.
            show_multi_modality_result(img,
                                       gt_bboxes,
                                       pred_bboxes,
                                       img_metas['lidar2img'],
                                       out_dir,
                                       filename,
                                       box_mode='lidar',
                                       img_metas=img_metas,
                                       show=False)

            # file_name = osp.split()
        return None

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """

        new_results = []
        for i in range(len(results)):
            box_type = type(results[i]['boxes_3d'])
            boxes_3d = results[i]['boxes_3d'].tensor
            boxes_3d = box_type(torch.cat(
                (boxes_3d, boxes_3d.new_zeros(
                    (boxes_3d.shape[0], 2))), dim=-1),
                                box_dim=9)
            new_results.append(
                dict(boxes_3d=boxes_3d,
                     scores_3d=results[i]['scores_3d'],
                     labels_3d=results[i]['labels_3d']))
        results = new_results
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')

        CAM_NUM = 6

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):

            if sample_id % CAM_NUM == 0:
                boxes_per_frame = []
                attrs_per_frame = []
            cam_index = sample_id % CAM_NUM
            cam_index = list(self.data_infos[sample_id //
                                             6]['cams'].keys())[cam_index]

            # need to merge results from images of the same sample
            annos = []
            cam_info = self.data_infos[sample_id // 6]['cams'][cam_index]
            boxes = mono_output_to_nusc_box(
                det,
                cam_info,
            )
            sample_token = self.data_infos[sample_id // 6]['token']

            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id // 6],
                boxes,
                mapped_class_names,
                self.eval_detection_configs,
                self.eval_version,
            )

            boxes_per_frame.extend(boxes)
            attrs = [torch.ones(len(boxes)) * 8
                     ]  # 8 means None, check in the attribute dicts
            attrs_per_frame.extend(attrs)
            # Remove redundant predictions caused by overlap of images
            if (sample_id + 1) % CAM_NUM != 0:
                continue

            boxes = global_nusc_box_to_cam(self.data_infos[sample_id // 6],
                                           boxes_per_frame, mapped_class_names,
                                           self.eval_detection_configs,
                                           cam_index, self.eval_version)
            cam_boxes3d, scores, labels = nusc_box_to_cam_box3d(boxes)
            # box nms 3d over 6 images in a frame
            # TODO: move this global setting into config
            nms_cfg = dict(use_rotate_nms=True,
                           nms_across_levels=False,
                           nms_pre=4096,
                           nms_thr=0.4,
                           score_thr=0.,
                           min_bbox_size=0,
                           max_per_frame=500)
            from mmcv import Config
            nms_cfg = Config(nms_cfg)
            cam_boxes3d_for_nms = xywhr2xyxyr(cam_boxes3d.bev)
            boxes3d = cam_boxes3d.tensor
            # generate attr scores from attr labels
            if len(attrs_per_frame) > 0:
                attrs = torch.cat(attrs_per_frame, dim=0)
            else:
                attrs = labels.new_tensor([8 for label in labels])
            if len(attrs) != len(boxes3d):
                import pdb; pdb.set_trace()
            boxes3d, scores, labels, attrs = box3d_multiclass_nms(
                boxes3d,
                cam_boxes3d_for_nms,
                scores,
                nms_cfg.score_thr,
                nms_cfg.max_per_frame,
                nms_cfg,
                mlvl_attr_scores=attrs)
            cam_boxes3d = CameraInstance3DBoxes(boxes3d, box_dim=9)
            det = bbox3d2result(cam_boxes3d, scores, labels, attrs)
            boxes, attrs = mono_output_to_nusc_box(det,
                                                   adjust_yawv1=False,
                                                   rotate=False,
                                                   convert_dim=False)
            boxes, attrs = cam_nusc_box_to_global(
                self.data_infos[sample_id // 6], boxes, attrs,
                mapped_class_names, self.eval_detection_configs, cam_index,
                self.eval_version)

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                # attr = self.get_attr_name(attrs[i], name)
                attr = self.get_attr_name(8, name)
                nusc_anno = dict(sample_token=sample_token,
                                 translation=box.center.tolist(),
                                 size=box.wlh.tolist(),
                                 rotation=box.orientation.elements.tolist(),
                                 velocity=box.velocity[:2].tolist(),
                                 detection_name=name,
                                 detection_score=box.score,
                                 attribute_name=attr)
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def get_cam_yaw(self, box, lidar2ego_translation, lidar2ego_rotation,
                    cam_info):
        yaws = []
        for box_idx in box:

            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_idx[6])
            nusc_box = NuScenesBox(box_idx[:3].numpy(),
                                   box_idx[3:6][[1, 0, 2]].numpy(),
                                   quat,
                                   label=0,
                                   score=1,
                                   velocity=[0, 0, 0])
            nusc_box.rotate(pyquaternion.Quaternion(lidar2ego_rotation))
            nusc_box.translate(np.array(lidar2ego_translation))
            nusc_box.translate(-np.array(cam_info['sensor2ego_translation']))
            nusc_box.rotate(
                pyquaternion.Quaternion(
                    cam_info['sensor2ego_rotation']).inverse)

            # nusc_box.translate(-np.array(cam_info['sensor2lidar_translation']))
            # nusc_box.rotate(
            # pyquaternion.Quaternion(matrix=cam_info['sensor2lidar_rotation']).inverse)
            v = np.dot(nusc_box.rotation_matrix, np.array([1, 0, 0]))
            yaw = -np.arctan2(v[2], v[0])
            yaws.append(yaw)
            # yaw = nusc_box.rotation.to_list()
            # yaw = nusc_box

        return torch.tensor(yaws)


def global_nusc_box_to_cam(info,
                           boxes,
                           classes,
                           eval_configs,
                           cam_index,
                           eval_version='detection_cvpr_2019'):
    """Convert the box from global to camera coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.translate(-np.array(info['ego2global_translation']))
        box.rotate(
            pyquaternion.Quaternion(info['ego2global_rotation']).inverse)
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to camera coord system
        box.translate(
            -np.array(info['cams'][cam_index]['sensor2ego_translation']))
        box.rotate(
            pyquaternion.Quaternion(
                info['cams'][cam_index]['sensor2ego_rotation']).inverse)
        box_list.append(box)
    return box_list


def nusc_box_to_cam_box3d(boxes):
    """Convert boxes from :obj:`NuScenesBox` to :obj:`CameraInstance3DBoxes`.

    Args:
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.

    Returns:
        tuple (:obj:`CameraInstance3DBoxes` | torch.Tensor | torch.Tensor): \
            Converted 3D bounding boxes, scores and labels.
    """
    locs = torch.Tensor([b.center for b in boxes]).view(-1, 3)
    dims = torch.Tensor([b.wlh for b in boxes]).view(-1, 3)
    rots = torch.Tensor([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).view(-1, 1)
    velocity = torch.Tensor([b.velocity[:2] for b in boxes]).view(-1, 2)
    boxes_3d = torch.cat([locs, dims, rots, velocity], dim=1).cuda()
    cam_boxes3d = CameraInstance3DBoxes(boxes_3d,
                                        box_dim=9,
                                        origin=(0.5, 0.5, 0.5))
    scores = torch.Tensor([b.score for b in boxes]).cuda()
    labels = torch.LongTensor([b.label for b in boxes]).cuda()
    nms_scores = scores.new_zeros(scores.shape[0], 10 + 1)
    indices = labels.new_tensor(list(range(scores.shape[0])))
    nms_scores[indices, labels] = scores
    return cam_boxes3d, nms_scores, labels


def cam_nusc_box_to_global(info,
                           boxes,
                           attrs,
                           classes,
                           eval_configs,
                           cam_index,
                           eval_version='detection_cvpr_2019'):
    """Convert the box from camera to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    attr_list = []
    for (box, attr) in zip(boxes, attrs):
        # Move box to ego vehicle coord system
        box.rotate(
            pyquaternion.Quaternion(
                info['cams'][cam_index]['sensor2ego_rotation']))
        box.translate(
            np.array(info['cams'][cam_index]['sensor2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
        attr_list.append(attr)
    return box_list, attr_list


def mono_output_to_nusc_box(detection,
                            cam_info=None,
                            adjust_yawv1=False,
                            rotate=True,
                            convert_dim=True):
    """Different from the function in nuscene_dataset, this function deals with
    box with lidar coord and cam yaw.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """

    box3d = detection['boxes_3d']
    # box3d = box3d.convert_to(Box3DMode.CAM, rt=extrinsic)
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    attrs = None
    if 'attrs_3d' in detection:
        attrs = detection['attrs_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    if adjust_yawv1 is True:
        # print(1)
        box_yaw = box_yaw - np.pi
        # pass
    # if adjust_yawv2 is True:
    #     box_yaw = -box_yaw - np.pi / 2
    if convert_dim:
        nus_box_dims = box_dims[:, [0, 2, 1]]
    else:
        nus_box_dims = box_dims
    box_list = []
    for i in range(len(box3d)):
        if rotate is True:
            quat = pyquaternion.Quaternion(
                        axis=[0, 1, 0], angle=box_yaw[i]) * \
                pyquaternion.Quaternion(axis=(1, 0, 0), radians=np.pi / 2)
        else:
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])

        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(box_gravity_center[i],
                          nus_box_dims[i],
                          quat,
                          label=labels[i],
                          score=scores[i],
                          velocity=velocity)
        if rotate is True:
            box.rotate(
                pyquaternion.Quaternion(
                    matrix=cam_info['sensor2lidar_rotation']))
            box.translate(cam_info['sensor2lidar_translation'])
        box_list.append(box)
    if attrs is None:
        return box_list
    else:
        return box_list, attrs



def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    if "ids" in detection:
        ids = detection["ids"].numpy()
    else:
        ids = None

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        if ids is not None:
            box['tracking_id'] = str(ids[i])
        box_list.append(box)
    return box_list
