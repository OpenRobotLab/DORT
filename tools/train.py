# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import argparse
import copy
import os
import time
import warnings
from os import path as osp

import mmcv
import numpy as np
import torch
import datetime
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from torch.utils.tensorboard import SummaryWriter

try:
    from mmtrack import __version__ as mmtrack_version
except:
    print('not install mmtrack')
    mmtrack_version = None
import logging
import sys
from det3d.utils.misc import modify_file_client_args
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--noshow-gpu', action='store_true')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--no_timeout',
        action="store_true",
        default=False)
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--no_filter_logger', action="store_false", default=True)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument(
        '--ceph',
        action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def preprocess_logger(cfg, args, project_name):
    new_log_config = []
    # with_mlflow = False
    # str_cfg = str(cfg)
    # exp_name = "det3d"
    # print(str_cfg)
    # if "kitti" in str_cfg:
    #     exp_name = "kitti"
    # if "nusc" in str_cfg:
    #     exp_name = "Nusc"
    # if "waymo" in str_cfg:
    #     exp_name = "Waymo"
    for hook in cfg['log_config']['hooks']:
        # if hook['type'] == 'CustomMlflowLoggerHook':
        #     hook['run_name'] = project_name
        #     hook['log_model'] = False
        #     hook['exp_name'] = exp_name
        #     # hook['cfg'] = cfg
        #     with_mlflow=True
        if hook['type'] == 'WandbLoggerHook':
            hook['init_kwargs']['name'] = project_name
            #hook['init_kwargs']['settings'] = wandb.Settings(start_method='thread')
            if args.ceph:
                 continue
        new_log_config.append(hook)

    # if not with_mlflow:
    #     hook = {}
    #     hook['type'] = 'CustomMlflowLoggerHook'
    #     hook['run_name'] = project_name
    #     hook['log_model'] = False
    #     hook['exp_name'] = exp_name
    #     # hook['cfg'] = cfg
    #     new_log_config.append(hook)
    cfg['log_config']['hooks'] = new_log_config
    return cfg



def main():
    args = parse_args()
    os.environ['WANDB_API_KEY'] ='5ec36fbfdd87fa5c7f804c4fc3adc3cf792bc23b'

    cfg = Config.fromfile(args.config)
    if args.local_rank == 0 and args.noshow_gpu is False:
        os.system("nvidia-smi")

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if args.ceph:
        cfg = modify_file_client_args(cfg)


    print( torch.cuda.is_available())

    # import modules from plguin/xx, registry will be updated
    # if hasattr(cfg, 'plugin'):
        # if cfg.plugin:
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

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    timestamp = time.strftime('%m%d_%H%M', time.localtime())

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        work_dir = osp.join(args.work_dir, osp.splitext(osp.basename(args.config))[0], timestamp)
        project_name = osp.join(osp.splitext(osp.basename(args.config))[0], timestamp)
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0], timestamp)
        project_name = osp.join(osp.splitext(osp.basename(args.config))[0], timestamp)


    ops_string = ''
    if args.cfg_options is not None:
        for key, item in args.cfg_options.items():
            ops_string += f'{key}:{item}-'
    # project_name = osp.basename(args.config) + "-" + ops_string + timestamp
    work_dir = work_dir + '-' + ops_string
    project_name = project_name + '-' + ops_string
    cfg.work_dir = work_dir

    cfg = preprocess_logger(cfg, args, project_name)

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    print(f'cfg gpu {cfg.gpu_ids}')


    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        if args.no_timeout:
            distributed = True
            init_dist(args.launcher,
                **cfg.dist_params)
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)
        else:
            distributed = True
            init_dist(args.launcher,
                 timeout=datetime.timedelta(seconds=18000),
                 **cfg.dist_params)
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model

    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level)

    # add a logging filter
    logging_filter = logging.Filter('mmdet')
    logging_filter.filter = lambda record: record.find('mmdet') != -1


    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged

    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    temp_data = datasets[0].__getitem__(0)
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            mmtrack_version=mmtrack_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    cfg.device="cuda"
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork', force=True)
    main()
