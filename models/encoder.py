"""Slice encoders behind one interface: [N,3,H,W] in [0,1] -> [N,D]."""
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Encoder(nn.Module):
    out_dim: int

    def __init__(self, freeze_norm=True):
        super().__init__()
        self.freeze_norm = freeze_norm
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def features(self, x):
        raise NotImplementedError

    def last_stage(self):
        """Module whose output feeds Grad-CAM."""
        raise NotImplementedError

    def forward(self, x):
        return self.features((x - self.mean) / self.std)

    def _freeze_bn(self):
        for mod in self.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.eval()
                for prm in mod.parameters():
                    prm.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_norm:
            self._freeze_bn()
        return self


class VisionHOPEEncoder(Encoder):
    """VisionHOPE backbone; ``forward_features`` returns the pooled 1024-d pre-classifier feature."""

    def __init__(self, name="visionhope_tiny", checkpoint=None, grad_checkpointing=True, freeze_norm=True):
        super().__init__(freeze_norm)
        from visionhope.models.factory import create_model
        self.net = create_model(name, checkpoint_path=checkpoint)
        self.net.head = nn.Identity()
        self.net.set_grad_checkpointing(grad_checkpointing)
        self.out_dim = self.net.num_features
        if freeze_norm:
            self._freeze_bn()

    def features(self, x):
        return self.net.forward_features(x)

    def last_stage(self):
        return self.net.stages[-1]


class TimmEncoder(Encoder):
    def __init__(self, name="convnext_tiny", grad_checkpointing=True, freeze_norm=True, pretrained=True):
        super().__init__(freeze_norm)
        import timm
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=0)
        if grad_checkpointing:
            self.net.set_grad_checkpointing(True)
        self.out_dim = self.net.num_features
        if freeze_norm:
            self._freeze_bn()

    def features(self, x):
        return self.net(x)

    def last_stage(self):
        return self.net.stages[-1] if hasattr(self.net, "stages") else self.net.blocks[-1]


def build_encoder(cfg):
    e = cfg["encoder"]
    if e["name"].startswith("visionhope"):
        from utils.config import ROOT
        ckpt = str(ROOT / e["checkpoint"]) if e.get("checkpoint") else None
        return VisionHOPEEncoder(e["name"], ckpt, e["grad_checkpointing"], e["freeze_norm"])
    return TimmEncoder(e["name"], e["grad_checkpointing"], e["freeze_norm"])


@torch.no_grad()
def encode_bag(encoder, x, chunk=64):
    """Eval-mode feature extraction for one bag in chunks; returns float32 [N,D]."""
    out = []
    for i in range(0, len(x), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(encoder(x[i:i + chunk]).float())
    return torch.cat(out)
