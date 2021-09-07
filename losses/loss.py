import torch
from torch import nn

from OhemCELoss import OhemCELoss
from detail_loss import DetailAggregateLoss


class Loss(nn.Module):
    def __init__(self, cfg, score_thresh=0.7, ignore_idx=255):
        super(Loss, self).__init__()

        self.cfg = cfg
        self.n_min = cfg.batch_size * cfg.crop_size[0] * cfg.crop_size[1] // 16
        self.criteria_p = OhemCELoss(thresh=score_thresh, n_min=self.n_min, ignore_lb=ignore_idx)
        self.criteria_16 = OhemCELoss(thresh=score_thresh, n_min=self.n_min, ignore_lb=ignore_idx)
        self.criteria_32 = OhemCELoss(thresh=score_thresh, n_min=self.n_min, ignore_lb=ignore_idx)
        self.boundary_loss_func = DetailAggregateLoss()

    def forward(self, lb, out, out16, out32, detail2=None, detail4=None, detail8=None):
        lossp = self.criteria_p(out, lb)
        loss2 = self.criteria_16(out16, lb)
        loss3 = self.criteria_32(out32, lb)

        boundary_bce_loss = 0.
        boundary_dice_loss = 0.

        if detail2 is not None:
            boundary_bce_loss2, boundary_dice_loss2 = self.boundary_loss_func(detail2, lb)
            boundary_bce_loss += boundary_bce_loss2
            boundary_dice_loss += boundary_dice_loss2

        if detail4 is not None:
            boundary_bce_loss4, boundary_dice_loss4 = self.boundary_loss_func(detail4, lb)
            boundary_bce_loss += boundary_bce_loss4
            boundary_dice_loss += boundary_dice_loss4

        if detail8 is not None:
            boundary_bce_loss8, boundary_dice_loss8 = self.boundary_loss_func(detail8, lb)
            boundary_bce_loss += boundary_bce_loss8
            boundary_dice_loss += boundary_dice_loss8

        loss = lossp + loss2 + loss3 + boundary_bce_loss + boundary_dice_loss
        return loss, boundary_bce_loss, boundary_dice_loss
