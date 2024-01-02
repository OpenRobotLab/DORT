import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .centernet_utils import gather_feature


def depth_uncertainty_loss(output, uncertainty, mask, index, target, uncertainty_weight=1):
    pred = gather_feature(output, index, use_transform=True)
    uncertainty = gather_feature(uncertainty, index, use_transform=True)
    mask = mask.unsqueeze(dim=2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask.detach(), target * mask.detach(), reduction="none")
    uncertainty[~mask.bool()]*= 0
    loss = loss * torch.exp(- uncertainty) + uncertainty * uncertainty_weight
    loss = loss.sum() / (mask.sum() + 1e-4)
    return loss


# for CenterTrack3D
def reg_l1_loss(output, mask, index, target, reduction="sum"):
    pred = gather_feature(output, index, use_transform=True)
    mask = mask.unsqueeze(dim=2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, reduction=reduction)
    loss = loss / (mask.sum() + 1e-4)
    return loss



class FocalLoss(nn.Module):

    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target, ignore=None):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, self.beta)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        # convert to 1e-6 due to the mixed precision issue
        prediction = torch.clamp(prediction, 1e-6)
        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) * torch.pow(prediction,
                                                              self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        if ignore is None:
            positive_loss = positive_loss.sum()
            negative_loss = negative_loss.sum()
        else:
            ignore = ignore.max(dim=1,keepdim=True)[0].expand_as(positive_loss)
            ignore[ignore!=0] = 1
            ignore = ignore.bool()
            num_positive = positive_index[~ignore].float().sum()
            positive_loss = positive_loss[~ignore].sum()
            negative_loss = negative_loss[~ignore].sum()
        if num_positive == 0:
            loss = -negative_loss
        else:
            loss = -(positive_loss + negative_loss) / num_positive

        return loss




class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, index, rotbin, rotres):
        pred = gather_feature(output, index, use_transform=True)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss

    def forward_retinanet(self, output, rotbin, rotes):
        pred = output
        loss = compute_rot_loss(pred, rotbin, rotes, mask=None)
        return loss

def compute_res_loss(output, target, mask=None):
    return F.smooth_l1_loss(output, target, reduction='mean')


def compute_bin_loss(output, target, mask):
    if mask is not None:
        # mask = mask.expand_as(output)
        output = output * mask.expand_as(output).float()
        mask = mask.bool().reshape(-1)
        return F.cross_entropy(output[mask], target[mask], reduction='mean')
    else:
        return F.cross_entropy(output, target, reduction='mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    if mask is not None:
        mask = mask.view(-1, 1)
    if mask.sum() == 0:
        loss_bin1 = output.new_zeros(1)
        loss_bin2 = output.new_zeros(1)
    else:
        loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0].long(), mask)
        loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1].long(), mask)
    loss_res = torch.zeros_like(loss_bin1)
    if torch.isnan(loss_bin1).sum() > 0:
        import pdb; pdb.set_trace()
    if torch.isnan(loss_bin2).sum() > 0:
        import pdb; pdb.set_trace()
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):
    #
    # orid = [sin(delta), cos(delta)] shape = [batch, bins, 2]
    # angleDiff = GT - center, shape = [batch, bins]
    #

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()
