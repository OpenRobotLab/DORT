# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import numpy as np
import warnings
from mmcv import Config, DictAction, mkdir_or_exist
from os import path as osp
from pathlib import Path
import os

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                               DepthInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.core.visualizer import (show_multi_modality_result, show_result,)
from mmdet3d.datasets import build_dataset
import sys
sys.path.insert(0, "../../")
from det3d.core.visualizer.show_result import show_custom_multi_modality_result
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['Normalize', 'NormalizeMultiviewImage'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument(
        '--task',
        type=str,
        choices=['det', 'seg', 'multi_modality-det', 'mono-det', 'bev-det'],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--aug',
        action='store_true',
        help='Whether to visualize augmented datasets or original dataset.')
    parser.add_argument(
        '--online',
        action='store_true',
        help='Whether to perform online visualization. Note that you often '
        'need a monitor to do so.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def build_data_cfg(config_path, skip_type, aug, cfg_options):
    """Build data config for loading visualization data."""

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # extract inner dataset of `RepeatDataset` as `cfg.data.train`
    # so we don't need to worry about it later
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    if cfg.data.train['type'] == 'RepeatDataset':
        cfg.data.train = cfg.data.train.dataset
    # use only first dataset for `ConcatDataset`
    if cfg.data.train['type'] == 'ConcatDataset':
        cfg.data.train = cfg.data.train.datasets[0]
    train_data_cfg = cfg.data.train

    if aug:
        show_pipeline = cfg.train_pipeline
    else:
        show_pipeline = cfg.eval_pipeline
        for i in range(len(cfg.train_pipeline)):
            if cfg.train_pipeline[i]['type'] == 'LoadAnnotations3D':
                show_pipeline.insert(i, cfg.train_pipeline[i])
    train_data_cfg['pipeline'] = [
        x for x in show_pipeline if x['type'] not in skip_type
    ]

    for idx, x in enumerate(train_data_cfg['pipeline']):
        if x['type'] == 'MultiViewPipeline':
            transform = [y for y in x['transforms'] if y['type'] not in skip_type]
            train_data_cfg['pipeline'][idx]['transforms'] = transform

    return cfg


def to_depth_mode(points, bboxes):
    """Convert points and bboxes to Depth Coord and Depth Box mode."""
    if points is not None:
        points = Coord3DMode.convert_point(points.copy(), Coord3DMode.LIDAR,
                                           Coord3DMode.DEPTH)
    if bboxes is not None:
        bboxes = Box3DMode.convert(bboxes.clone(), Box3DMode.LIDAR,
                                   Box3DMode.DEPTH)
    return points, bboxes


def show_det_data(input, out_dir, show=False):
    """Visualize 3D point cloud and 3D bboxes."""
    img_metas = input['img_metas']._data
    if "points" in input:
        points = input['points']._data.numpy()
    else:
        points = None
    gt_bboxes = input['gt_bboxes_3d']._data.tensor
    if img_metas['box_mode_3d'] != Box3DMode.DEPTH:
        points, gt_bboxes = to_depth_mode(points, gt_bboxes)
    filename = osp.splitext(osp.basename(img_metas['pts_filename']))[0]
    show_result(
        points,
        gt_bboxes.clone(),
        None,
        out_dir,
        filename,
        show=show,
        snapshot=True)


def show_seg_data(input, out_dir, show=False):
    """Visualize 3D point cloud and segmentation mask."""
    img_metas = input['img_metas']._data
    points = input['points']._data.numpy()
    gt_seg = input['pts_semantic_mask']._data.numpy()
    filename = osp.splitext(osp.basename(img_metas['pts_filename']))[0]
    show_seg_result(
        points,
        gt_seg.copy(),
        None,
        out_dir,
        filename,
        np.array(img_metas['PALETTE']),
        img_metas['ignore_index'],
        show=show,
        snapshot=True)


def show_proj_bbox_img(input, out_dir, show=False, is_nus_mono=False, vis_task="bev-det"):
    """Visualize 3D bboxes on 2D image by projection."""
    gt_bboxes = input['gt_bboxes_3d']._data
    img_metas = input['img_metas']._data
    img = input['img']._data.numpy()
    # need to transpose channel to first dim
    if vis_task != "bev-det":
        if len(img.shape) == 4:
            img = img[0]
        img = img.transpose(1, 2, 0)
    # no 3D gt bboxes, just show img
    if gt_bboxes.tensor.shape[0] == 0:
        gt_bboxes = None
    if isinstance(img_metas['filename'], list):
        filename = Path(img_metas['filename'][0]).name
        cam_idx = re.findall("image_([0-9])", img_metas['filename'][0])
        if len(cam_idx) > 0:
            filename = filename + "_" + cam_idx[0]
    else:
        filename = Path(img_metas['filename']).name
        cam_idx = re.findall("image_([0-9])", img_metas['filename'])
        if len(cam_idx) > 0:
            filename =  filename + "_" + cam_idx
    if vis_task == "bev-det":
        show_custom_multi_modality_result(
            img,
            gt_bboxes,
            None,
            img_metas,
            out_dir,
            filename,
            box_mode="lidar",
            img_metas=img_metas,
            show=show)

    elif isinstance(gt_bboxes, DepthInstance3DBoxes):
        show_multi_modality_result(
            img,
            gt_bboxes,
            None,
            None,
            out_dir,
            filename,
            box_mode='depth',
            img_metas=img_metas,
            show=show)
    elif isinstance(gt_bboxes, LiDARInstance3DBoxes):
        show_multi_modality_result(
            img,
            gt_bboxes,
            None,
            img_metas['lidar2img'],
            out_dir,
            filename,
            box_mode='lidar',
            img_metas=img_metas,
            show=show)
    elif isinstance(gt_bboxes, CameraInstance3DBoxes):
        show_multi_modality_result(
            img,
            gt_bboxes,
            None,
            img_metas['cam2img'],
            out_dir,
            filename,
            box_mode='camera',
            img_metas=img_metas,
            show=show)
    else:
        # can't project, just show img
        warnings.warn(
            f'unrecognized gt box type {type(gt_bboxes)}, only show image')
        show_multi_modality_result(
            img, None, None, None, out_dir, filename, show=show)


def main():
    args = parse_args()

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    cfg = build_data_cfg(args.config, args.skip_type, args.aug,
                         args.cfg_options)
    try:
        dataset = build_dataset(
            cfg.data.train, default_args=dict(filter_empty_gt=False))
    except TypeError:  # seg dataset doesn't have `filter_empty_gt` key
        dataset = build_dataset(cfg.data.train)

    dataset_type = cfg.dataset_type
    # configure visualization mode
    vis_task = args.task  # 'det', 'seg', 'multi_modality-det', 'mono-det', 'bev-det'
    progress_bar = mmcv.ProgressBar(len(dataset))

    for input in dataset:
        if vis_task in ['det', 'multi_modality-det']:
            # show 3D bboxes on 3D point clouds
            show_det_data(input, args.output_dir, show=args.online)
        if vis_task in ['multi_modality-det', 'mono-det', 'bev-det']:
            # project 3D bboxes to 2D image
            show_proj_bbox_img(
                input,
                args.output_dir,
                show=args.online,
                is_nus_mono=(dataset_type == 'NuScenesMonoDataset'),
                vis_task=vis_task)
        elif vis_task in ['seg']:
            # show 3D segmentation mask on 3D point clouds
            show_seg_data(input, args.output_dir, show=args.online)
        progress_bar.update()


if __name__ == '__main__':
    main()
