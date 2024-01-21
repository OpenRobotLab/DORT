from det3d.datasets.nuscenes_dataset import CustomNuScenesDataset

# from .dataset_wrappers import MultiViewMixin
import numpy as np

from mmdet3d.core.bbox import CameraInstance3DBoxes

from .nuscenes_dataset import CustomNuScenesDataset
from mmdet.datasets.api_wrappers import COCO
import copy
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class CustomNuScenesMonoFrameDataset(CustomNuScenesDataset):
    r"""NuScenes Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    ref_img_sampler for tracking dataset, default is None,
    an example of ref_img_sampler:
        ref_img_sampler=dict(
            frame_range=[-3, 3],
            stride=1,
            num_ref_imgs=1)
    """

    def __init__(self,
                cam_ann_file=None,
                **kwargs):
        super().__init__(**kwargs)
        self.cam_ann_file = cam_ann_file
        self.cam_data_infos = \
                self.load_cam_annotations(cam_ann_file)
        
        assert len(self.cam_data_infos) == \
                                6 * len(self.data_infos)
        
    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def load_cam_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
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


    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]


    def get_data_info(self, index):
        # ori_index = index
        """if self.empty_mask[index] is True:

        return None
        """
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
        import pdb; pdb.set_trace()
        return input_dict