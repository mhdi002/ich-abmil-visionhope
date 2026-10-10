"""Gated attention MIL (Ilse et al., 2018) with per-class or shared attention."""
import torch
from torch import nn


def masked_softmax(scores, mask, dim=-1):
    """scores [B,C,N], mask [B,N] or per-slot [B,C,N] (True = valid slice)."""
    m = mask[:, None, :] if mask.dim() == 2 else mask
    scores = scores.masked_fill(~m, float("-inf"))
    return torch.softmax(scores.float(), dim=dim)


class SliceContext(nn.Module):
    """Residual SRNL over the slice sequence (ablation). Requires CUDA."""

    def __init__(self, dim, head_dim=16):
        super().__init__()
        from visionhope.models.srnl import SRNL
        self.srnl = SRNL(dim=dim, head_dim=head_dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, mask):
        z = z * mask[..., None]
        return z + self.norm(self.srnl(z.float().contiguous())).to(z.dtype)


class GatedAttentionMIL(nn.Module):
    """Per-slot gated attention pooling. Slots = classes, then binary tasks, then regression tasks
    (multi-task); the first ``n_classes`` slots are the hemorrhage classes and behave as before."""

    def __init__(self, in_dim, n_classes, proj_dim=256, attn_dim=128, dropout=0.25,
                 attention="per_class", srnl_context=False, aux=True, n_tasks=0, n_reg=0, slot_groups=None,
                 uncertainty=False):
        super().__init__()
        # learned loss weights (Kendall et al., 2018) for [classes + aux, binary tasks, regression]
        self.log_vars = nn.Parameter(torch.zeros(3)) if uncertainty else None
        self.n_classes, self.n_tasks, self.n_reg = n_classes, n_tasks, n_reg
        n_slots = n_classes + n_tasks + n_reg
        self.n_slots = n_slots
        # slice group each slot may attend to (0 = brain bag, 1 = basal-cistern bag; dual-bag models only)
        self.register_buffer("slot_groups", torch.as_tensor(slot_groups if slot_groups is not None else [0] * n_slots,
                                                           dtype=torch.long), persistent=False)
        self.attention = attention
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.context = SliceContext(proj_dim) if srnl_context else None
        self.attn_v = nn.Linear(proj_dim, attn_dim)
        self.attn_u = nn.Linear(proj_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, n_slots if attention == "per_class" else 1)
        self.drop = nn.Dropout(dropout)
        self.cls_w = nn.Parameter(torch.randn(n_slots, proj_dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(n_slots))
        self.aux = nn.Linear(proj_dim, 1) if aux else None

    def forward(self, h, mask, grp=None):
        z = self.proj(h)
        if self.context is not None:
            z = self.context(z, mask)
        gate = torch.tanh(self.attn_v(z)) * torch.sigmoid(self.attn_u(z))
        scores = self.attn_w(gate).transpose(1, 2)                     # [B,C|1,N]
        if scores.shape[1] == 1:
            scores = scores.expand(-1, self.n_slots, -1)
        if grp is not None and bool((self.slot_groups != 0).any()):
            valid = mask[:, None, :] & (grp[:, None, :] == self.slot_groups[None, :, None])   # [B,S,N]
            empty = ~valid.any(-1, keepdim=True)                       # a group missing in a scan: use all slices
            mask = torch.where(empty, mask[:, None, :].expand_as(valid), valid)
        attn = masked_softmax(scores, mask)                           # [B,S,N]
        pooled = torch.einsum("bcn,bnd->bcd", attn.to(z.dtype), z)    # [B,S,D]
        pooled = self.drop(pooled)
        slots = (pooled * self.cls_w).sum(-1) + self.cls_b           # [B,S]
        C, K = self.n_classes, self.n_tasks
        out = {"logits": slots[:, :C], "attn": attn[:, :C]}
        if K or self.n_reg:
            out.update(task_logits=slots[:, C:C + K], reg=slots[:, C + K:], task_attn=attn[:, C:])
        if self.aux is not None:
            out["aux_logit"] = self.aux(pooled[:, :C].mean(1)).squeeze(-1)
        if self.log_vars is not None:
            out["log_vars"] = self.log_vars
        return out


class PoolMIL(nn.Module):
    """Mean pooling (embedding-level) or max pooling (instance-level, mi-Net style)."""

    def __init__(self, in_dim, n_classes, proj_dim=256, dropout=0.25, mode="mean", aux=True):
        super().__init__()
        self.mode = mode
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(proj_dim, n_classes)
        self.aux = nn.Linear(proj_dim, 1) if aux else None

    def forward(self, h, mask):
        z = self.proj(h)
        m = mask[..., None].to(z.dtype)
        if self.mode == "mean":
            pooled = (z * m).sum(1) / m.sum(1).clamp(min=1)
            logits = self.cls(pooled)
            attn = (mask.float() / mask.float().sum(1, keepdim=True))[:, None].expand(-1, logits.shape[1], -1)
            out = {"logits": logits, "attn": attn}
            agg = pooled
        else:
            inst = self.cls(z).masked_fill(~mask[..., None], float("-inf"))   # [B,N,C]
            logits = inst.max(1).values
            attn = torch.softmax(inst.float().transpose(1, 2), -1)
            out = {"logits": logits, "attn": attn}
            agg = z.masked_fill(~mask[..., None], float("-inf")).max(1).values
        if self.aux is not None:
            out["aux_logit"] = self.aux(agg).squeeze(-1)
        return out


def build_head(cfg, in_dim):
    from data.dataset import task_names
    mc = cfg["model"]
    n = len(cfg["classes"])
    tb, tr = task_names(cfg)
    groups = None
    if mc.get("dual_bag"):
        cist = set((cfg.get("tasks") or {}).get("cistern_slots") or [])
        groups = [1 if name in cist else 0 for name in list(cfg["classes"]) + tb + tr]
    if mc["pooling"] == "abmil":
        return GatedAttentionMIL(in_dim, n, mc["proj_dim"], mc["attn_dim"], mc["dropout"],
                                 mc["attention"], mc["srnl_context"], n_tasks=len(tb), n_reg=len(tr),
                                 slot_groups=groups,
                                 uncertainty=bool((tb or tr) and (cfg.get("tasks") or {}).get("uncertainty_weighting")))
    assert not (tb or tr), "multi-task heads need model.pooling=abmil"
    return PoolMIL(in_dim, n, mc["proj_dim"], mc["dropout"], mc["pooling"])


def load_head_partial(model, state, log=print):
    """Load a checkpoint whose head has fewer slots (e.g. 5-class model -> multi-task model).

    Tensors with the same shape are copied; per-slot tensors (attn_w, cls_w, cls_b) get their first rows copied
    and keep the fresh initialization for the new slots.
    """
    own = model.state_dict()
    partial = []
    for k, v in state.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k].copy_(v)
        elif own[k].dim() == v.dim() and own[k].shape[1:] == v.shape[1:] and own[k].shape[0] > v.shape[0]:
            own[k][:v.shape[0]].copy_(v)
            partial.append(k)
        else:
            raise ValueError(f"cannot map {k}: {tuple(v.shape)} -> {tuple(own[k].shape)}")
    missing = [k for k in own if k not in state]
    log(f"partial load: {len(partial)} per-slot tensors extended {partial}, {len(missing)} new tensors {missing[:6]}")


class BagModel(nn.Module):
    """Encoder + MIL head for end-to-end training on all slices of each bag."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x, mask, grp=None):
        """x [B,N,3,H,W], mask [B,N], grp [B,N] slice group (dual-bag). Only valid slices are encoded."""
        B, N = mask.shape
        feats = self.encoder(x[mask])
        h = feats.new_zeros(B, N, feats.shape[-1])
        h[mask] = feats
        return self.head(h, mask) if grp is None else self.head(h, mask, grp)
