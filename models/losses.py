import torch
import torch.nn.functional as F
from torch import nn


class AsymmetricLoss(nn.Module):
    """Ridnik et al., 2021. Optional per-class pos_weight scales the positive term."""

    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05, pos_weight=None, eps=1e-8):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, y):
        p = torch.sigmoid(logits.float())
        p_neg = (p - self.clip).clamp(min=0) if self.clip else p
        lp = y * torch.log(p.clamp(min=self.eps)) * (1 - p).pow(self.gp)
        if self.pos_weight is not None:
            lp = lp * self.pos_weight
        ln = (1 - y) * torch.log((1 - p_neg).clamp(min=self.eps)) * p_neg.pow(self.gn)
        return -(lp + ln).sum(-1).mean()


class FocalBCE(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, y):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight, reduction="none")
        pt = torch.exp(-F.binary_cross_entropy_with_logits(logits, y, reduction="none"))
        return ((1 - pt).pow(self.gamma) * bce).sum(-1).mean()


class BCE(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=self.pos_weight,
                                                  reduction="none").sum(-1).mean()


def build_loss(cfg, train_labels):
    lc = cfg["loss"]
    pw = None
    if lc["use_pos_weight"]:
        y = torch.as_tensor(train_labels, dtype=torch.float32)
        pos = y.sum(0).clamp(min=1)
        pw = torch.sqrt((len(y) - pos) / pos)
    if lc["name"] == "asl":
        # ASL already down-weights easy negatives; pos_weight is applied only for focal/bce.
        return AsymmetricLoss(lc["asl_gamma_neg"], lc["asl_gamma_pos"], lc["asl_clip"])
    if lc["name"] == "focal":
        return FocalBCE(lc["focal_gamma"], pw)
    return BCE(pw)
