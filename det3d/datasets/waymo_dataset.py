import torch
import numpy as np
from mmdet.datasets import DATASETS

from mmdet3d.datasets import WaymoDataset
from .kitti_dataset import CustomKittiDataset
import os
import mmcv
@DATASETS.register_module()
class CustomWaymoDataset(WaymoDataset, CustomKittiDataset):
    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix="velodyne",
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d="Lidar",
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 modify_yaw_offset=np.pi,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 multiview_index=["image_0",
                                "image_1", "image_2",
                                "image_3", "image_4", ],):

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            load_interval=load_interval,
            pcd_limit_range=pcd_limit_range)

        self.multiview_index = multiview_index

        self.default_multiview_index = "image_0"

        self.modify_yaw_offset = modify_yaw_offset



    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        # should check the img_metas

        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)

        lidar2cam = rect @ Trv2c
        images_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam2imgs = []

        for multiview_index_idx in self.multiview_index:
            images_paths.append(img_filename.replace(
                        self.default_multiview_index, multiview_index_idx))
            calib_idx = multiview_index_idx.replace("image_", "P")
            cam2img_idx = info['calib'][calib_idx]
            cam2imgs.append(cam2img_idx)



            lidar2img_rts.append(cam2img_idx @ lidar2cam)
            lidar2cam_rts.append(lidar2cam)

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=images_paths),
            img_filename=images_paths,
            lidar2img=lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam2img=cam2imgs,)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def evaluate(self,
                 results,
                 metric='kitti',
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        '''
            modify the default metric from waymo to kitti
        '''

        return super().evaluate(
                    results,
                    metric,
                    logger,
                    pklfile_prefix,
                    submission_prefix,
                    show,
                    out_dir,
                    pipeline)



@DATASETS.register_module()
class CustomMonoWaymoDataset(CustomWaymoDataset):

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 modify_yaw_offset=np.pi,
                 load_interval=1,
                 pre_filter_empty_gt=True,
                 multiview_index = ["image_0",
                                "image_1", "image_2",
                                "image_3", "image_4"],
                 pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0]):
        # if augmentation with stereo image,
        # the multiview index can be ["image_2", "image_3"]
        self.n_views = len(multiview_index)
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            modify_yaw_offset=modify_yaw_offset,
            multiview_index = multiview_index,
            load_interval=load_interval,
            pcd_limit_range=pcd_limit_range)

        self.n_views = len(multiview_index)


        if self.test_mode is False and pre_filter_empty_gt is True:
            self.empty_mask = self.filter_empty_gt_data()
        else:
            self.empty_mask = torch.ones(self.__len__())



    def __len__(self):
        return  len(self.data_infos) * self.n_views

    def get_data_info(self, index):
        ori_index = index

        if self.empty_mask[index] is True:
            return None
        multiview_index_idx = index % self.n_views
        multiview_index_idx = self.multiview_index[multiview_index_idx]
        index = index // self.n_views

        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']

        img_filename = os.path.join(self.data_root,
                                info['image']['image_path'])


        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)

        lidar2cam = rect @ Trv2c

        images_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam2imgs = []

        images_paths.append(img_filename.replace(
                    self.default_multiview_index, multiview_index_idx))
        calib_idx = multiview_index_idx.replace("image_", "P")
        cam2img_idx = info['calib'][calib_idx]
        cam2imgs.append(cam2img_idx)



        lidar2img_rts.append(cam2img_idx @ lidar2cam)
        lidar2cam_rts.append(lidar2cam)


        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=images_paths),
            img_filename=images_paths,
            lidar2img=lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam2img=cam2imgs,)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        #annos = self.modify_yaw_angle(input_dict, annos)

        return input_dict

    def modify_yaw_angle(self, img_metas, annos):
        # get corners
        corners = annos['gt_bboxes']
        # convert corners to cam index

        # get corresponding yaw angle

        # get yaw angle
        pass

    def _format_bbox(self, results, josnfile_prefix=None):
        return super()._format_bbox(self, results, jsonfile_prefix)

    def filter_empty_gt_data(self):
        return torch.ones(self.__len__())
        # new_data_infos = []
        # new_ann_infos = []
        # empty_mask = []
        # empty_num = 0
        # # Adopt pickle processing
        # pkl_name = self.ann_file.replace(".pkl", "") + "-gt_mask.pkl"
        # if osp.exists(pkl_name):
        #     empty_mask, empty_num = mmcv.load(pkl_name)
        # else:
        #     for idx in tqdm(range(len(self))):
        #         input_dict = self.get_data_info(idx)
        #         ann_info = input_dict['ann_info']

        #         lidar2cam = input_dict['lidar2cam'][0]
        #         lidar2img = input_dict['lidar2img'][0]

        #         gt_bboxes = ann_info['gt_bboxes_3d'].convert_to(
        #                         Box3DMode.CAM, torch.tensor(lidar2cam))

        #         projected_center = gt_bboxes.projected_gravity_center(torch.tensor(intrinsic))

        #         mask = (projected_center[:, 0] > 0) & (projected_center[:, 1] > 0) & \
        #             (projected_center[:, 0] < 1920) & (projected_center[:, 1] < 1280) & \
        #                 (gt_bboxes.tensor[:, 2] > 0)

        #         if mask.sum() > 0:
        #             empty_mask.append(False)
        #         else:
        #             empty_num += 1
        #             empty_mask.append(True)
        #     mmcv.dump([empty_mask, empty_num], pkl_name)
        # print(f"num of example {len(self)}, empty num {empty_num}")
        # return empty_mask


    def filter_invalid_bbox(self, input_dict, ann_info):
        lidar2cam = input_dict['lidar2cam'][0]
        lidar2img = input_dict['lidar2img'][0]
        projected_center = gt_bboxes.projected_gravity_center(torch.tensor(intrinsic))

        mask = (projected_center[:, 0] > 0) & (projected_center[:, 1] > 0) & \
            (projected_center[:, 0] < 1920) & (projected_center[:, 1] < 1280) & \
                (gt_bboxes.tensor[:, 2] > 0)

        ann_info = ann_info[mask]
        return ann_info


    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert results to kitti format for evaluation and test submission.

        Args:
            net_outputs (List[np.ndarray]): list of array storing the
                bbox and score
            class_nanes (List[String]): A list of class names
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            List[dict]: A list of dict have the kitti 3d format
        """
        assert len(net_outputs) == len(self.data_infos) * self.n_views, \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        # conflict with idx
        for img_idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            if img_idx % self.n_views == 0:
                sum_pred_dicts = pred_dicts
            else:

                sum_pred_dicts['boxes_3d'].tensor = \
                    torch.cat([sum_pred_dicts['boxes_3d'].tensor,
                                 pred_dicts['boxes_3d'].tensor], dim=0)
                sum_pred_dicts['scores_3d'] = \
                    torch.cat([sum_pred_dicts['scores_3d'],
                                 pred_dicts['scores_3d']], dim=0)
                sum_pred_dicts['labels_3d'] = \
                    torch.cat([sum_pred_dicts['labels_3d'],
                                 pred_dicts['labels_3d']], dim=0)

            if (img_idx + 1) % self.n_views == 0:
                annos = []
                frame_idx = img_idx // self.n_views
                info = self.data_infos[frame_idx]
                sample_idx = info['image']['image_idx']
                image_shape = info['image']['image_shape'][:2]

                box_dict = self.convert_valid_bboxes(sum_pred_dicts, info)

                # TODO should do the nms operation
                if len(box_dict['bbox']) > 0:
                    box_2d_preds = box_dict['bbox']
                    box_preds = box_dict['box3d_camera']
                    scores = box_dict['scores']
                    box_preds_lidar = box_dict['box3d_lidar']
                    label_preds = box_dict['label_preds']

                    anno = {
                        'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'dimensions': [],
                        'location': [],
                        'rotation_y': [],
                        'score': []
                    }

                    for box, box_lidar, bbox, score, label in zip(
                            box_preds, box_preds_lidar, box_2d_preds, scores,
                            label_preds):
                        bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                        bbox[:2] = np.maximum(bbox[:2], [0, 0])
                        anno['name'].append(class_names[int(label)])
                        anno['truncated'].append(0.0)
                        anno['occluded'].append(0)
                        anno['alpha'].append(
                            -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                        anno['bbox'].append(bbox)
                        anno['dimensions'].append(box[3:6])
                        anno['location'].append(box[:3])
                        anno['rotation_y'].append(box[6])
                        anno['score'].append(score)

                    anno = {k: np.stack(v) for k, v in anno.items()}
                    annos.append(anno)

                    if submission_prefix is not None:
                        curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                        with open(curr_file, 'w') as f:
                            bbox = anno['bbox']
                            loc = anno['location']
                            dims = anno['dimensions']  # lhw -> hwl

                            for idx in range(len(bbox)):
                                print(
                                    '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                    '{:.4f} {:.4f} {:.4f} '
                                    '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.
                                    format(anno['name'][idx], anno['alpha'][idx],
                                        bbox[idx][0], bbox[idx][1],
                                        bbox[idx][2], bbox[idx][3],
                                        dims[idx][1], dims[idx][2],
                                        dims[idx][0], loc[idx][0], loc[idx][1],
                                        loc[idx][2], anno['rotation_y'][idx],
                                        anno['score'][idx]),
                                    file=f)
                else:
                    annos.append({
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    })
                annos[-1]['sample_idx'] = np.array(
                    [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

                det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos



    def convert_valid_bboxes(self, box_dict, info):
        """
        Add the offset of yaw angle
        """
        if self.modify_yaw_offset!=0:
            box_dict['boxes_3d'].tensor[:, 6] += self.modify_yaw_offset

        return super().convert_valid_bboxes(box_dict, info)

