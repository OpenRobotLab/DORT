import os.path as osp
from pathlib import Path

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
import tempfile
from mmdet3d.datasets import NuScenesMonoDataset
from torch.utils.data import Dataset
from mmdet3d.datasets import Custom3DDataset
from .nuscenes_dataset import CustomNuScenesDataset
from mmdet3d.core.bbox import get_box_type
import warnings
from mmdet3d.datasets.pipelines import Compose
import copy

@DATASETS.register_module()
class NuScenesBevDetDataset(CustomNuScenesDataset, NuScenesMonoDataset):
    r"""NuScenes Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    """

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
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 speed_mode='relative_dis',
                 frame_num = 2,
                 max_interval=3,
                 min_interval=0,
                 prev_only=False,
                 next_only=False,
                 test_adj = 'prev',
                 ref_img_sampler=None,
                 dataset_mode = "detection",
                 fix_direction=False,
                 test_adj_ids=None,
                 load_mono_anno = False,
                 mono_ann_file = None,
                 load_interval_shuffle=1,
                 pkl_version="v1.0",
                 multi_adj_frame_id_cfg=(1, 9, 1),
                 file_client_args=dict(backend='disk')):
        self.NUM_IMGS = 6
        self.pkl_version = pkl_version
        self.frame_num = frame_num
        self.ref_img_sampler = ref_img_sampler
        self.load_interval = load_interval
        self.dataset_mode = dataset_mode
        self.load_interval_shuffle = load_interval_shuffle
        self.use_valid_flag = use_valid_flag
        Dataset.__init__(self)
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
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
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
                use_external=False,
            )

        self.img_info_prototype = img_info_prototype

        self.speed_mode = speed_mode
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.prev_only = prev_only
        self.next_only = next_only
        self.test_adj = test_adj
        self.fix_direction = fix_direction
        self.test_adj_ids = test_adj_ids

        self.load_mono_anno = load_mono_anno
        if self.load_mono_anno:
            self.mono_data_infos = self.load_mono_annotations(mono_ann_file)

            assert len(self.mono_data_infos) == \
                                    6 * len(self.data_infos)


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


    def get_cam_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_cam_ann_info(self.data_infos[idx], ann_info)

    def _parse_cam_ann_info(self, img_info, ann_info):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, labels,
                gt_bboxes_3d, gt_labels_3d, attr_labels, centers2d,
                depths, bboxes_ignore, masks, seg_map
        """
        gt_bboxes = []
        gt_labels = []
        attr_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                attr_labels.append(ann['attribute_id'])
                gt_masks_ann.append(ann.get('segmentation', None))
                # 3D annotations in camera coordinates
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
                velo_cam3d = np.array(ann['velo_cam3d']).reshape(1, 2)
                nan_mask = np.isnan(velo_cam3d[:, 0])
                velo_cam3d[nan_mask] = [0.0, 0.0]
                bbox_cam3d = np.concatenate([bbox_cam3d, velo_cam3d], axis=-1)
                gt_bboxes_cam3d.append(bbox_cam3d.squeeze())
                # 2.5D annotations in camera coordinates
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            attr_labels = np.array(attr_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            attr_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 0.5, 0.5))
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_labels_3d=gt_labels_3d,
            attr_labels=attr_labels,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

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
        # standard protocal modified from SECOND.Pytorch
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
        for key in ["lidar2ego_translation",
                    "lidar2ego_rotation",
                    "ego2global_translation",
                    "ego2global_rotation"]:
            input_dict[key] = info[key]


        if self.modality['use_camera']:
            cam2img = []
            for cam_type, cam_info in info['cams'].items():
                intrinsic = cam_info['cam_intrinsic']
                cam2img.append(intrinsic)

            input_dict.update(
                dict(cam2img=cam2img))
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
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

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))
            elif self.img_info_prototype == 'bevdet':
                input_dict.update(dict(img_info=info['cams']))
            elif self.img_info_prototype == 'bevdet_sequential':                
                if info ['prev'] is None or info['next'] is None:
                    adjacent= 'prev' if info['next'] is None else 'next'
                else:
                    if self.prev_only or self.next_only:
                        adjacent = 'prev' if self.prev_only else 'next'
                    elif self.test_mode:
                        adjacent = self.test_adj
                    else:
                        adjacent = np.random.choice(['prev', 'next'])
                if type(info[adjacent]) is list:
                    if self.test_mode:
                        if self.test_adj_ids is not None:
                            info_adj=[]
                            select_id = self.test_adj_ids
                            for id_tmp in select_id:
                                id_tmp = min(id_tmp, len(info[adjacent])-1)
                                info_adj.append(info[adjacent][id_tmp])
                        else:
                            select_id = min((self.max_interval+self.min_interval)//2,
                                            len(info[adjacent])-1)
                            if self.frame_num == 2:
                                info_adj = info[adjacent][select_id]
                            else:
                                info_adj = []
                                for i in range(self.frame_num - 1):
                                    select_idx = select_id * i
                                    if select_idx > len(info[adjacent]) -1:
                                        select_idx = select_id
                                    info_adj.append(info[adjacent][select_idx])
                    else:
                        if len(info[adjacent])<= self.min_interval:
                            if self.frame_num == 2:
                                select_id = len(info[adjacent])-1
                            else:
                                select_id = []
                                select_id0 = (len(info[adjacent]) - 1) // (self.frame_num - 1)
                                for i in range(self.frame_num - 1):
                                    select_id.append(
                                        select_id0*i)
                        else:
                            if self.frame_num == 2:
                                select_id = np.random.choice([adj_id for adj_id in range(
                                    min(self.min_interval,len(info[adjacent])),
                                    min(self.max_interval,len(info[adjacent])))])
                            else:
                                select_id =  []
                                max_idx = len(info[adjacent]) // (self.frame_num-1)
                                if max_idx > min(self.min_interval,len(info[adjacent])):
                                    select_id0 = np.random.choice([adj_id for adj_id in range(
                                        min(self.min_interval,len(info[adjacent])),
                                        min(self.max_interval,max_idx))])
                                else:
                                    select_id0 = min(self.min_interval,len(info[adjacent]))
                                for i in range(self.frame_num - 1):
                                    select_id_idx = min(select_id0 * i, len(info[adjacent]) - 1)
                                    select_id.append(select_id_idx)
                        if self.frame_num == 2:
                            info_adj = info[adjacent][select_id]
                        else:
                            info_adj = [info[adjacent][select_id_idx] for select_id_idx in select_id]
                else:
                    if self.frame_num == 2:
                        info_adj = info[adjacent]
                    else:
                        info_adj = []
                        for i in range(self.frame_num - 1):
                            info_adj.append(info[adjacent])
                input_dict.update(dict(img_info=info['cams'],
                                       curr=info,
                                       adjacent=info_adj,
                                       adjacent_type=adjacent))
            elif self.img_info_prototype == 'bevdet_sequentialv2':
                info_adj_list = self.get_adj_info(info, index)
                input_dict.update(dict(img_info=info['cams'],
                                       adjacent=info_adj_list,
                                       curr=info))
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.img_info_prototype == 'bevdet_sequential':
                bbox = input_dict['ann_info']['gt_bboxes_3d'].tensor
                if 'abs' in self.speed_mode:
                    bbox[:, 7:9] = bbox[:, 7:9] + torch.from_numpy(info['velo']).view(1,2).to(bbox)
                if input_dict['adjacent_type'] == 'next' and not self.fix_direction:
                    bbox[:, 7:9] = -bbox[:, 7:9]
                if 'dis' in self.speed_mode:
                    if self.frame_num == 2:
                        time = abs(input_dict['timestamp'] - 1e-6 * input_dict['adjacent']['timestamp'])
                        bbox[:, 7:9] = bbox[:, 7:9] * time
                    else:
                        time = abs(input_dict['timestamp'] - 1e-6 * input_dict['adjacent'][-1]['timestamp'])
                        bbox[:, 7:9] = bbox[:, 7:9] * time
                input_dict['ann_info']['gt_bboxes_3d'] = LiDARInstance3DBoxes(bbox,
                                                                              box_dim=bbox.shape[-1],
                                                                              origin=(0.5, 0.5, 0.0))
            if self.load_mono_anno:
                mono_annos = []
                for i in range(self.NUM_IMGS):
                    mono_annos.append(
                        self.get_mono_ann_info(index*self.NUM_IMGS + i))
                input_dict['mono_ann_info'] = mono_annos
        return input_dict


    def get_adj_info(self, info, index):
        info_adj_list = []
        for select_id in range(*self.multi_adj_frame_id_cfg):
            if select_id == 0:
                continue
            select_id = max(index - select_id, 0)
            select_id = min(select_id, len(self) - 1)
            # mean that the prev info is not from the same scene.

            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def get_mono_ann_info(self, index):
        img_id = self.mono_data_infos[index]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.mono_data_infos[index], ann_info)

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
            boxes = output_to_nusc_box(det, self.data_infos[sample_id],
                                       self.speed_mode, self.img_info_prototype,
                                       self.max_interval, self.test_adj,
                                       self.fix_direction,
                                       self.test_adj_ids,
                                       self.pkl_version)
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
                    nusc_anno['tracking_name'] = nusc_anno['detection_name']
                    nusc_anno['tracking_score'] = nusc_anno['detection_score']
                    nusc_anno.pop('detection_name')
                    nusc_anno.pop('detection_score')
                    nusc_anno.pop('attribute_name')
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


def output_to_nusc_box(detection, info, speed_mode,
                       img_info_prototype, max_interval, test_adj, fix_direction,
                       test_adj_ids, pkl_version="v1.0"):
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

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    if pkl_version == "v1.0":
        nus_box_dims = box_dims[:, [1, 0, 2]]
    else:
        box_yaw = -box_yaw - np.pi / 2
        nus_box_dims = box_dims[:,]

    velocity_all = box3d.tensor[:, 7:9]
    if img_info_prototype =='bevdet_sequential':
        if info['prev'] is None or info['next'] is None:
            adjacent = 'prev' if info['next'] is None else 'next'
        else:
            adjacent = test_adj
        if adjacent == 'next' and not fix_direction:
            velocity_all = -velocity_all
        if type(info[adjacent]) is list:
            select_id = min(max_interval // 2, len(info[adjacent]) - 1)
            # select_id = min(2, len(info[adjacent]) - 1)
            info_adj = info[adjacent][select_id]
        else:
            info_adj = info[adjacent]
        if 'dis' in speed_mode and test_adj_ids is None:
            time = abs(1e-6 * info['timestamp'] - 1e-6 * info_adj['timestamp'])
            velocity_all = velocity_all / time
    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*velocity_all[i,:], 0.0)
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
        box_list.append(box)
    return box_list
