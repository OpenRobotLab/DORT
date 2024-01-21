plugin=True
plugin_dir='det3d/'

_base_ = ["../../bevdet/bevdet_stt_tiny.py"]
data_root = 'data/nuscenes/'
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (384, 1056),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test':0.04,
}

# Model
grid_config={
        'xbound': [-51.2, 51.2, 0.8],
        'ybound': [-51.2, 51.2, 0.8],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 1.0],}

voxel_size = [0.1, 0.1, 0.2]



numC_Trans=64

model = dict(
    type='CustomSORT',
    _delete_=True,
    # ['bicycle', 'car', 'bus', 'motorcycle', 'pedestrian', 'trailer', 'truck']
    keep_classes=[0, 1, 3, 4, 6, 7, 8],
    detector=dict(
        type='CustomImVoxelNet',
        use_grid_mask=False,
        bev_det_format=True,
        select_first_neck_feat = False,
        init_cfg=dict(type='Pretrained',
                      checkpoint='./ckpts/bevdet_stt_tiny_cbgs_res1056x384_0709_1009_epoch20.pth'),
        backbone=dict(
            type='CustomSwinTransformer',
            pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
            pretrain_img_size=224,
            embed_dims=96,
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            strides=(4, 2, 2, 2),
            out_indices=(2, 3,),
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.0,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN', requires_grad=True),
            pretrain_style='official',
            output_missing_index_as_none=False),
        neck=dict(
            type='FPN_LSS',
            in_channels=384+768,
            out_channels=512,
            extra_upsample=None,
            input_feature_index=(0,1),
            scale_factor=2),
        neck_bev=dict(
            type='ResNetForBEVDet',
            numC_input=numC_Trans,
            neck=dict(
                type='FPN_LSS',
                in_channels=numC_Trans*8+numC_Trans*2,
                out_channels=256)),
        view_transform=dict(type='ViewTransformerLiftSplatShoot',
                            grid_config=grid_config,
                            data_config=data_config,
                            numC_Trans=numC_Trans),
        bbox_head=dict(
            type='CustomCenterHead',
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
                pc_range=point_cloud_range[:2],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=9),
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3),
            loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            norm_bbox=True),
        train_cfg=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]),
        test_cfg=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            # nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            # nms_thr=0.2,

            # Scale-NMS
            nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]]
            )),
    tracker=dict(
        type='CustomSORTTracker'),
    motion=dict(
        type='AB3DMOTKalmanFilter'))

# Data
dataset_type = 'NuScenesBevDetDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
    
train_pipeline = [
    dict(type='BevDetLoadMultiViewImageFromFiles',
        is_train=True,
        data_config=data_config,
        file_client_args=file_client_args),
    dict(
        type='BevDetLoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='BevDetGlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='BevDetRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                                    'gt_instances_id'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'img_info'))
]

eval_pipeline = [
    dict(type='BevDetLoadMultiViewImageFromFiles',
        data_config=data_config,
        file_client_args=file_client_args),
    # # load lidar points for --show in test.py only
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     file_client_args=file_client_args),
    dict(type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
    dict(type='CustomCollect3D', keys=['img_inputs'])
]
train_post_pipeline = [
    dict(type='CustomMatchInstances'),
    dict(type='SeqFormating'),
]

test_pipeline = [
    dict(type='BevDetLoadMultiViewImageFromFiles',
        data_config=data_config,
        file_client_args=file_client_args),
    # # load lidar points for --show in test.py only
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     file_client_args=file_client_args),
    dict(type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
    dict(type='CustomCollect3D', keys=['img_inputs'])
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
            ann_file=data_root + 'nuscenes_mini_infos_val.pkl',
            pipeline=test_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=True,
            eval_version=eval_version,
            box_type_3d='LiDAR'),)
load_from=''
