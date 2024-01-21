import mmcv
import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses import L1Loss, l1_loss
# from .utils import weighted_loss

@LOSSES.register_module()
class UncertaintyL1Loss(nn.Module):
    """L1 loss with modeling laplacian distribution

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean',
                       loss_weight=1.0, 
                       uncertainty_weight=1.0,
                       uncertainty_range=[-10, 10]):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.uncertainty_weight = uncertainty_weight
        self.uncertainty_range = uncertainty_range
    
    def forward(self, pred, target, uncertainty,
                     weight=None, avg_factor=None,
                     reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = l1_loss(
            pred, target, weight, reduction="none",)
        uncertainty_clamp = uncertainty.clamp(-10, 10)

        if loss.dim() ==2:
            loss = loss.mean(dim=1)

        loss = loss * torch.exp(- uncertainty_clamp) + \
                        uncertainty_clamp * self.uncertainty_weight
        loss = self.loss_weight * loss.mean()
        return loss