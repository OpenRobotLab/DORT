_base_ = [
    './hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py'
]
voxel_size = [0.32, 0.32, 6]
model = dict(
    type='MVXFasterRCNN',
    pts_bbox_head=dict(
        type='Anchor3DHead',
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345],
                    [-74.88, -74.88, 0, 74.88, 74.88, 0],
                    [-74.88, -74.88, -0.1188, 74.88, 74.88, -0.1188]],
            sizes=[
                [4.73, 2.08, 1.77],  # car
                [0.91, 0.84, 1.74],  # pedestrian
                [1.81, 0.84, 1.77],  # cyclist

            ],
            rotations=[0, 1.57],
            reshape_out=False)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(  # car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # pedestrian
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                dict(  # cyclist
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False)))
