plugin=True
plugin_dir='det3d/'

_base_ = ['../../monovoxel/monovoxel_r101_1x8_nuscenes_centerhead_aug_pretrained.py']



n_voxels=[312, 312, 12]

model = dict(
    type='CustomSORT',
    _delete_=True,
    # ['bicycle', 'car', 'bus', 'motorcycle', 'pedestrian', 'trailer', 'truck']
    keep_classes=[0, 1, 2, 3, 5, 6, 7],
    detector=dict(
        type='CustomImVoxelNet',
        use_grid_mask=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint= # noqa: E251
            './ckpts/imvoxelnet_track_debug.pth'),
        backbone=dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
            style='pytorch'),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=64,
            num_outs=4),
        view_transform=dict(type='LSSImVoxelViewTransform',
                        n_voxels=n_voxels,
                        anchor_generator=dict(
                            type='AlignedAnchor3DRangeGenerator',
                            ranges=[[-49.6, -49.6, -2.92, 49.6, 49.6, 0.92]],
                            rotations=[.0])),
        neck_3d=dict(type='NuScenesImVoxelNeck', in_channels=64, out_channels=256),
        bbox_head=dict(
            type='CenterHead',
            in_channels=256,
            tasks=[
                dict(num_class=1, class_names=['car']),
                dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                dict(num_class=2, class_names=['bus', 'trailer']),
                dict(num_class=1, class_names=['barrier']),
                dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),],
            common_heads=dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
                    share_conv_channel=64,
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                post_center_range=[-49.92, -49.92, -2.92, 49.92, 49.92, 0.92],
                # pc_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=[-49.92, -49.92, -2.92, 49.92, 49.92, 0.92],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=1,
                voxel_size=(.64, .64),
                code_size=9),
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3),
            loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            norm_bbox=True),
        n_voxels=[312, 312, 12],
        train_cfg = dict(
            grid_size=[156, 156, 1],
            voxel_size=(.64, .64, 3.84),
            out_size_factor=1,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range = [-49.92, -49.92, -2.92, 49.92, 49.92, 0.92]),
        test_cfg = dict(
            # post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_limit_range=[-49.92, -49.92, -2.92, 49.92, 49.92, 0.92],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=(.64, .64),
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2),),
    #track_head=dict(
    #   type='CustomQuasiDesneEmbedHead',
    #    mode='BEV'),
    tracker=dict(
        type='CustomSORTTracker'),
    motion=dict(
        type='AB3DMOTKalmanFilter'))


dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range =[-49.92, -49.92, -2.92, 49.92, 49.92, 0.92]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)



train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='CustomLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False, with_instance_ids=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='MultiViewRandomFlip3D', flip_ratio=0.5),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d',
                                       'gt_labels_3d',
                                       'gt_instance_ids',
                                       'img'])
]
train_post_pipeline = [
    dict(type='CustomMatchInstances'),
    dict(type='SeqFormating'),
]
test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img']),
]

eval_version="tracking_nips_2019"
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_mini_infos_train.pkl',
        pipeline=train_pipeline,
        post_pipeline=train_post_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        eval_version=eval_version,
        ref_img_sampler=dict(
            frame_range=[-3, 3],),
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type='ConcatDataset',
        datasets=[dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'nuscenes_mini_infos_train.pkl',
                pipeline=test_pipeline,
                modality=input_modality,
                classes=class_names,
                test_mode=True,
                eval_version=eval_version,
                box_type_3d='LiDAR'),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'nuscenes_mini_infos_val.pkl',
                pipeline=test_pipeline,
                modality=input_modality,
                classes=class_names,
                test_mode=True,
                eval_version=eval_version,
                box_type_3d='LiDAR'),],
        separate_eval=True),
    test=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_val.pkl',
            pipeline=test_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=True,
            eval_version=eval_version,
            box_type_3d='LiDAR'),)
load_from=''
