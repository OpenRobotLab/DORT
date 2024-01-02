# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)
from mmcv.parallel import DataContainer


def single_gpu_get_loss(model,
                        data_loader,
                        show=False,
                        out_dir=None,
                        show_score_thr=0.3):
    """
        Calculate the loss instead of getting the prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=True, rescale=True, **data)
        assert show==False
        results.append(result)

        batch_size = len(data)
        for _ in range(1): # the batch size is 1
            prog_bar.update()

    output_dicts = {}
    try:
        for idx, result in enumerate(results):
            for key, item in result.items():
                item = item.reshape(-1)
                if key not in output_dicts:
                    output_dicts[key] = item / len(data_loader)
                else:
                    output_dicts[key] += item / len(data_loader)
    except:
        import pdb; pdb.set_trace()
    for key, item in output_dicts.items():
        print("{}: {}".format(key, item))
    return output_dicts
