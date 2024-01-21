import numpy as np
import os
from mmdet.datasets import DATASETS
from mmdet3d.datasets import KittiDataset
from tqdm import tqdm

@DATASETS.register_module()
class CustomKittiDataset(KittiDataset):
    r"""Kitti Dataset.
    This dataset adds camera intrinsics and extrinsics to the results.
    and support stereo data.

    multiview_index: index for selecting the images in the multivew database.
    """

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
                 modify_yaw_offset=0,
                 remove_hard_instance_level=0,
                 load_prev_frame=False,
                 multiview_index = ["image_2", "image_3"],
                 pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0]):

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
            pcd_limit_range=pcd_limit_range)
        self.remove_hard_instance_level = remove_hard_instance_level

        self.multiview_index = multiview_index
        self.default_multiview_index = "image_2"

        self.modify_yaw_offset = modify_yaw_offset

        self.load_prev_frame = load_prev_frame

        self.n_views = len(multiview_index)
        if remove_hard_instance_level > 0:
            for idx, info in tqdm(enumerate(self.data_infos)):
                info['annos'] = self.remove_hard_instances(info['annos'])
                self.data_infos[idx] = info


    def remove_hard_instances(self, ann_info):
        if self.remove_hard_instance_level == 0:
            return ann_info
        elif self.remove_hard_instance_level == 1:
            # occluded >= 2
            # depth >= 60
            mask = (ann_info['occluded'] <=2) & \
                     (ann_info['location'][:, 2] <=60)
            for key, item in ann_info.items():
                ann_info[key] = item[mask]
        elif self.remove_hard_instance_level == 2:
            # 1.
            mask = (ann_info["location"][:, 2] < 80)
            mask = mask & (ann_info["location"][:, 2] > 0)
            mask = mask & (np.abs(ann_info["location"][:, 1]) < 40)
            truncated_mask = ann_info['truncated'] >=0.9
            truncated_mask = truncated_mask & \
                ((ann_info["bbox"][:,2:] - ann_info["bbox"][:,:2]).min(axis=1) <=20)

            mask = mask &  (~truncated_mask)
            # mask = mask & (ann_info['truncated'])
            for key, item in ann_info.items():
                ann_info[key] = item[mask]
        elif self.remove_hard_instance_level == 3:
            # 1.
            mask = (ann_info["location"][:, 2] < 60)
            mask = mask & (ann_info["location"][:, 2] > 0)
            mask = mask & (np.abs(ann_info["location"][:, 1]) < 40)
            truncated_mask = ann_info['truncated'] >=0.9
            truncated_mask = truncated_mask & \
                ((ann_info["bbox"][:,2:] - ann_info["bbox"][:,:2]).min(axis=1) <=20)

            mask = mask &  (~truncated_mask)
            # mask = mask & (ann_info['truncated'])
            for key, item in ann_info.items():
                ann_info[key] = item[mask]

        return ann_info

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P2 = info['calib']['P2'].astype(np.float32)
        lidar2cam = rect @ Trv2c

        images_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam2img = []
        # lidar2img = P2 @ rect @ Trv2c

        for multiview_index_idx in self.multiview_index:
            images_paths.append(img_filename.replace(
                        self.default_multiview_index, multiview_index_idx))
            if multiview_index_idx == "image_2":
                cam2img_idx = info['calib']['P2'].astype(np.float32)
            else:
                cam2img_idx = info['calib']['P3'].astype(np.float32)

            cam2img.append(cam2img_idx)

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
            cam2img=cam2img,)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict


    def convert_valid_bboxes(self, box_dict, info):
        """
        Add the offset of yaw angle
        """
        if self.modify_yaw_offset!=0:
            box_dict['boxes_3d'].tensor[:, 6] += self.modify_yaw_offset

        return super().convert_valid_bboxes(box_dict, info)




@DATASETS.register_module()
class CustomMonoKittiDataset(CustomKittiDataset):
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
                 modify_yaw_offset=0,
                 load_prev_frame=False,
                 remove_hard_instance_level=0,
                 multiview_index = ["image_2"],
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
            remove_hard_instance_level=remove_hard_instance_level,
            modify_yaw_offset=modify_yaw_offset,
            load_prev_frame=load_prev_frame,
            multiview_index = multiview_index,
            pcd_limit_range=pcd_limit_range)
    def __len__(self):
        return self.n_views * super().__len__()

    def get_data_info(self, index):
        view_index = index % self.n_views
        index = index // self.n_views
        input_dict = super().get_data_info(index)
        input_dict['img_filename'] = [input_dict['img_filename'][view_index]]
        input_dict['lidar2img'] = [input_dict['lidar2img'][view_index]]
        input_dict['lidar2cam'] = [input_dict['lidar2cam'][view_index]]
        input_dict['cam2img'] = [input_dict['cam2img'][view_index]]

        return input_dict


