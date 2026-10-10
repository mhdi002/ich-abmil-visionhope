"""Bag datasets (cached slices or cached features), padding collate, GPU augmentation, balanced sampler."""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from utils.config import p


def load_table(cfg, require_split=True):
    """Manifest rows that survived preprocessing, joined with splits."""
    m = pd.read_csv(p(cfg, "manifest"))
    m["key"] = m["accession"].astype(int).astype(str) + "_" + m["ct_index"].astype(int).astype(str)
    st = pd.read_csv(p(cfg, "outputs") / "preprocess_stats.csv")
    m = m[m["usable"]].merge(st, on="key", how="inner")
    m = m[m["ok"] == True]  # noqa: E712
    excl = p(cfg, "outputs") / "orientation" / "exclude.csv"
    if excl.exists():
        m = m[~m["key"].isin(pd.read_csv(excl)["key"])]
    if require_split:
        sp = pd.read_csv(p(cfg, "splits"))[["key", "split", "fold", "chrono_holdout"]]
        m = m.merge(sp, on="key", how="inner")
    return m.reset_index(drop=True)


def load_slices(cache_dir, key, retries=5):
    """Cached uint8 bag [N,3,S,S] for a scan: <key>.npy, or compressed <key>.npz (preprocess.cache_compress).

    Retries short / failed reads: WSL's drvfs mount of a Windows drive occasionally returns a truncated read under
    load (a 3.3 MB file read as 4 KB), which killed a 2-hour epoch once although the file on disk was intact.
    """
    import time
    for attempt in range(retries):
        try:
            f = cache_dir / f"{key}.npy"
            if f.exists():
                return np.load(f)
            with np.load(cache_dir / f"{key}.npz") as z:
                return z["x"]
        except (ValueError, OSError, EOFError):
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def task_names(cfg):
    """(binary task names, regression task names); empty unless tasks.enabled."""
    t = cfg.get("tasks") or {}
    if not t.get("enabled"):
        return [], []
    return list((t.get("binary") or {}).keys()), list((t.get("regression") or {}).keys())


def task_targets(df, cfg):
    """Binary [n,K] and scaled regression [n,M] targets from manifest columns (see tasks in the config)."""
    tb, tr = task_names(cfg)
    t = cfg.get("tasks") or {}
    T = np.stack([df.eval(t["binary"][k]).astype(np.float32).values for k in tb], 1) if tb else         np.zeros((len(df), 0), np.float32)
    R = np.stack([df[t["regression"][k][0]].astype(np.float32).values / float(t["regression"][k][1]) for k in tr], 1)         if tr else np.zeros((len(df), 0), np.float32)
    return T, R


class SliceBagDataset(Dataset):
    """Returns uint8 slices [N,3,S,S]; augmentation and resizing run on the GPU."""

    def __init__(self, df, cfg):
        self.keys = df["key"].tolist()
        self.y = df[cfg["classes"]].values.astype(np.float32)
        self.aux = df[cfg["aux_target"]].values.astype(np.float32)
        self.t, self.r = task_targets(df, cfg)
        self.dir = p(cfg, "slice_cache")
        # dual bag: brain slices (group 0) + basal-cistern slices (group 1) of the same scan, one forward pass
        self.dir2 = p(cfg, "cistern_cache") if cfg["model"].get("dual_bag") else None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        x = load_slices(self.dir, self.keys[i])
        grp = np.zeros(len(x), np.int64)
        if self.dir2 is not None:
            x2 = load_slices(self.dir2, self.keys[i])
            x, grp = np.concatenate([x, x2]), np.concatenate([grp, np.ones(len(x2), np.int64)])
        return dict(x=torch.from_numpy(x), y=torch.from_numpy(self.y[i]), aux=torch.tensor(self.aux[i]),
                    key=self.keys[i], t=torch.from_numpy(self.t[i]), r=torch.from_numpy(self.r[i]),
                    grp=torch.from_numpy(grp))


class FeatureBagDataset(Dataset):
    """Cached encoder features. ``views`` > 1 samples one augmented view per access during training."""

    def __init__(self, df, cfg, feat_dir, views=("clean",), train=False):
        self.keys = df["key"].tolist()
        self.y = df[cfg["classes"]].values.astype(np.float32)
        self.aux = df[cfg["aux_target"]].values.astype(np.float32)
        self.dir = feat_dir
        self.views = list(views)
        self.train = train

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        v = self.views[np.random.randint(len(self.views))] if self.train else self.views[0]
        h = torch.from_numpy(np.load(self.dir / v / f"{self.keys[i]}.npy").astype(np.float32))
        return dict(x=h, y=torch.from_numpy(self.y[i]), aux=torch.tensor(self.aux[i]), key=self.keys[i])


def pad_collate(batch):
    n = max(b["x"].shape[0] for b in batch)
    shape = batch[0]["x"].shape[1:]
    x = batch[0]["x"].new_zeros((len(batch), n, *shape))
    mask = torch.zeros(len(batch), n, dtype=torch.bool)
    for i, b in enumerate(batch):
        k = b["x"].shape[0]
        x[i, :k] = b["x"]
        mask[i, :k] = True
    out = dict(x=x, mask=mask, y=torch.stack([b["y"] for b in batch]),
               aux=torch.stack([b["aux"] for b in batch]), key=[b["key"] for b in batch])
    for k in ("t", "r"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    if "grp" in batch[0]:
        g = torch.full((len(batch), n), -1, dtype=torch.long)
        for i, b in enumerate(batch):
            g[i, :len(b["grp"])] = b["grp"]
        out["grp"] = g
    return out


def gpu_prepare(x_uint8, cfg, train, generator=None):
    """[N,3,S,S] uint8 on GPU -> float [N,3,size,size] in [0,1].

    Train: one random rotation/scale/translation per bag (slices share patient pose),
    mild contrast/brightness and Gaussian noise. Never flips (orientation is normalized).
    """
    size = cfg["encoder"]["input_size"]
    x = x_uint8.float() / 255.0
    if train:
        a = cfg["augment"]
        dev = x.device
        r = lambda lo, hi: float(torch.empty(1).uniform_(lo, hi, generator=generator))  # noqa: E731
        ang = math.radians(r(-a["rotate_deg"], a["rotate_deg"]))
        s = r(*a["scale"])
        tx, ty = r(-a["translate"], a["translate"]), r(-a["translate"], a["translate"])
        theta = torch.tensor([[math.cos(ang) / s, -math.sin(ang) / s, tx],
                              [math.sin(ang) / s, math.cos(ang) / s, ty]], device=dev, dtype=x.dtype)
        grid = F.affine_grid(theta.expand(len(x), 2, 3), (len(x), 3, size, size), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        c = 1.0 + r(-a["contrast"], a["contrast"])
        b = r(-a["brightness"], a["brightness"])
        x = ((x - 0.5) * c + 0.5 + b)
        x = x + a["noise_std"] * torch.randn(x.shape, device=dev, generator=None)
        x = x.clamp(0, 1)
    else:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return x


def balanced_sampler(labels, mix=0.5, seed=0):
    """Scan-level class-balanced sampling, mixed with uniform to limit over-repetition."""
    y = np.asarray(labels)
    n_pos = y.sum(0).clip(min=1)
    neg = y.sum(1) == 0
    w = (y / n_pos).max(1)
    w[neg] = 1.0 / max(neg.sum(), 1)
    w = w / w.sum()
    w = mix * w + (1 - mix) / len(y)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(y),
                                 replacement=True, generator=g)
