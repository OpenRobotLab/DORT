# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import sys
sys.path.append("../")

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from data_converter import nuscenes_converter as nuscenes_converter

def add_ann_adj_info(extra_tag, test=False):
    nuscenes_version = 'v1.0-trainval' if test is False else 'v1.0-test'
    dataroot = '../data/nuscenes/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    data_set= ['train', 'val'] if test is False else ['test']
    for set in data_set:
        dataset = pickle.load(
            open('../data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            if test is False:
                ann_infos = list()
                for ann in sample['anns']:
                    ann_info = nuscenes.get('sample_annotation', ann)
                    velocity = nuscenes.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    ann_infos.append(ann_info)
                dataset['infos'][id]['ann_infos'] = ann_infos
                # dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']
        with open('../data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == "__main__":
    root_path = '../data/nuscenes'
    # extra_tag = 'bevdetv2-nuscenes'
    extra_tag = 'nuscenes'
    # add_ann_adj_info(extra_tag, test=False)
    add_ann_adj_info(extra_tag, test=False)
