import numpy as np
import torch
import torch.nn.functional as F


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap




class CenterNetHeatMap(object):
    @staticmethod
    def generate_scoremap(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = CenterNetHeatMap.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            CenterNetHeatMap.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])


    @staticmethod
    def generate_depth_map(dmap, gt_wh, centers_int, locations):
        dmap.fill_(10000)
        w, h = dmap.shape[1:]
        for idx in range(gt_wh.shape[0]):
            dpatch = CenterNetHeatMap.assign_depth_patch(gt_wh[idx], centers_int[idx], locations[idx])
            offset = (gt_wh[idx,:]/2).int()
            x1 = max(0, centers_int[idx, 1] - offset[1])
            x2 = min(w, centers_int[idx, 1] + offset[1])
            y1 = max(0, centers_int[idx, 0] - offset[0])
            y2 = min(h, centers_int[idx, 0] + offset[0])
            dmap[0,x1:x2, y1:y2] = torch.min(dmap[0,x1:x2, y1:y2], dpatch[:x2-x1,:y2-y1])
        dmap[dmap==10000] = 0


    @staticmethod
    def assign_depth_patch(gt_wh, center_int, location):
        gt_wh = gt_wh.int()
        dpatch = torch.zeros(gt_wh[1], gt_wh[0]).fill_(location[-1])
        return dpatch

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        if box_size.device.type == "cpu":
            box_tensor = torch.Tensor(box_size)
            width, height = box_tensor[..., 0], box_tensor[..., 1]

        else:
            width, height = box_size[..., 0], box_size[..., 1]

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m: m + 1, -n: n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CenterNetHeatMap.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian).to(fmap.device)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top: y + bottom, x - left: x + right]
        masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top: y + bottom, x - left: x + right] = masked_fmap
        # return fmap


class CenterNetDecoder(object):
    @staticmethod
    def pseudo_nms(fmap, pool_size=3):
        r"""
        apply max pooling to get the same effect of nms

        Args:
            fmap(Tensor): output tensor of previous step
            pool_size(int): size of max-pooling
        """
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == fmap).float()
        return fmap * keep

    @staticmethod
    def topk_score(scores, K=40):
        """
        get top K point in score map
        """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds.float() / width).int().float()
        topk_xs = (topk_inds.float() % width).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index.float() / K).int()
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs




def decode_dimension(dimension, cls_id, dim_mean, dim_mode):
    cls_id = cls_id.max(dim=1)[1]
    cls_dimension_mean = dim_mean[cls_id, :]

    if cls_dimension_mean.dim() > 3:
        cls_dimension_mean = cls_dimension_mean.permute(0, 3, 1, 2)
    if dim_mode[0] == "exp":
        dimension = dimension.exp()
    if dim_mode[2]:
        raise NotImplementedError
    else:
        dimension = dimension * cls_dimension_mean
    return dimension
