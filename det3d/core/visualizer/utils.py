
from matplotlib.cm import get_cmap
import os.path as osp
import mmcv
import os
import numpy as np
def colorise(tensor, cmap="coolwarm", vmin=None, vmax=None):
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    tensor = tensor.detach().cpu().float()
    zeros_mask = tensor <= 0.0001
    vmin = float(tensor.min()) if vmin is None else vmin
    vmax = float(tensor.max()) if vmax is None else vmax

    tensor = (tensor - vmin) / (vmax - vmin)

    output = cmap(tensor.detach().numpy())[..., :3]
    return output




def draw_nocs_img(input_meta, raw_img, nocs_img, seg_mask,  root="work_dirs/vis_nocs"):
    filename = input_meta['filename']
    if isinstance(filename, list):
        filename = filename[0]
    basename = osp.basename(filename).replace(".png", "")

    mean = input_meta['img_norm_cfg']['mean']
    std = input_meta['img_norm_cfg']['std']
    to_rgb = input_meta['img_norm_cfg']['to_rgb']
    os.makedirs(osp.join(root, basename), exist_ok=True)

    for idx, (raw_img_idx, nocs_img_idx, seg_mask_idx) in enumerate(zip(raw_img, nocs_img, seg_mask)):
        raw_img_idx = raw_img_idx.cpu().permute(1, 2, 0).numpy()
        raw_img_idx = mmcv.imdenormalize(raw_img_idx, mean, std, to_bgr=to_rgb).astype(np.uint8)

        seg_mask_idx = colorise(seg_mask_idx)[0]
        mmcv.imwrite(
            seg_mask_idx,
            osp.join(root, basename, f"{idx}_segmask.png"))
        mmcv.imwrite(
            raw_img_idx,
            osp.join(root, basename, f"{idx}_raw.png"),)
        nocs_img_idx = nocs_img_idx.cpu().permute(1, 2, 0).numpy()
        nocs_img_idx = mmcv.imdenormalize(nocs_img_idx, mean, std, to_bgr=to_rgb).astype(np.uint8)
        mmcv.imwrite(
            nocs_img_idx,
            osp.join(root, basename, f"{idx}_nocs.png"),)
