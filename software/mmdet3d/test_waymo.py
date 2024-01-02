import tensorflow as tf
import tempfile
from mmcv.utils import print_log
import mmcv
import os.path as osp
from mmdet3d.core.evaluation.waymo_utils.prediction_kitti_to_waymo import KITTI2Waymo
prefix = '1'
waymo_root = "data/waymo/waymo_format"
waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
result_files = mmcv.load("/mnt/lustre/lianqing/data/work_dirs/pp_waymo_v1/pts_bbox.pkl")
save_tmp_dir = tempfile.TemporaryDirectory()
waymo_results_save_dir = save_tmp_dir.name
pklfile_prefix = "/mnt/lustre/lianqing/data/work_dirs/pp_waymo_v1/"
waymo_results_final_path = f'{pklfile_prefix}.bin'
print(waymo_results_final_path)
print(save_tmp_dir.name)

converter = KITTI2Waymo(result_files, waymo_tfrecords_dir,
  waymo_results_save_dir,  waymo_results_final_path, prefix)
converter.convert()
save_tmp_dir.cleanup()
import subprocess
ret_bytes = subprocess.check_output(
    'mmdet3d/core/evaluation/waymo_utils/' +
    f'compute_detection_metrics_main {pklfile_prefix}.bin ' +
    f'{waymo_root}/gt.bin',
    shell=True)
ret_texts = ret_bytes.decode('utf-8')
print_log(ret_texts)
# parse the text to get ap_dict
ap_dict = {
    'Vehicle/L1 mAP': 0,
    'Vehicle/L1 mAPH': 0,
    'Vehicle/L2 mAP': 0,
    'Vehicle/L2 mAPH': 0,
    'Pedestrian/L1 mAP': 0,
    'Pedestrian/L1 mAPH': 0,
    'Pedestrian/L2 mAP': 0,
    'Pedestrian/L2 mAPH': 0,
    'Sign/L1 mAP': 0,
    'Sign/L1 mAPH': 0,
    'Sign/L2 mAP': 0,
    'Sign/L2 mAPH': 0,
    'Cyclist/L1 mAP': 0,
    'Cyclist/L1 mAPH': 0,
    'Cyclist/L2 mAP': 0,
    'Cyclist/L2 mAPH': 0,
    'Overall/L1 mAP': 0,
    'Overall/L1 mAPH': 0,
    'Overall/L2 mAP': 0,
    'Overall/L2 mAPH': 0
}
mAP_splits = ret_texts.split('mAP ')
mAPH_splits = ret_texts.split('mAPH ')
for idx, key in enumerate(ap_dict.keys()):
    split_idx = int(idx / 2) + 1
    if idx % 2 == 0:  # mAP
        ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
    else:  # mAPH
        ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
ap_dict['Overall/L1 mAP'] = \
    (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
        ap_dict['Cyclist/L1 mAP']) / 3
ap_dict['Overall/L1 mAPH'] = \
    (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
        ap_dict['Cyclist/L1 mAPH']) / 3
ap_dict['Overall/L2 mAP'] = \
    (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
        ap_dict['Cyclist/L2 mAP']) / 3
ap_dict['Overall/L2 mAPH'] = \
    (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
        ap_dict['Cyclist/L2 mAPH']) / 3
