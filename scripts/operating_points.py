"""Recall / specificity trade-off of a Stage A checkpoint on val (cached features, CPU)."""
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from data.dataset import FeatureBagDataset, load_table, pad_collate
from models.abmil import build_head
from utils.config import load_config, p

run = sys.argv[1] if len(sys.argv) > 1 else "abmil_vh"
cfg = load_config(None)
ck = torch.load(p(cfg, "outputs") / "runs" / run / "best.pt", map_location="cpu", weights_only=False)
head = build_head(ck["cfg"], ck["in_dim"])
head.load_state_dict(ck["head"])
head.eval()
classes = cfg["classes"]
df = load_table(cfg)
fdir = p(cfg, "feature_cache") / cfg["encoder"]["name"]
dl = DataLoader(FeatureBagDataset(df[df["split"] == "val"], cfg, fdir, ["clean"]), batch_size=64, collate_fn=pad_collate)
Y, P = [], []
with torch.no_grad():
    for b in dl:
        Y.append(b["y"].numpy())
        P.append(torch.sigmoid(head(b["x"].float(), b["mask"])["logits"].float()).numpy())
Y, P = np.concatenate(Y), np.concatenate(P)

print(f"run {run} (epoch {ck['epoch']}), val n={len(Y)}")
print(f"{'class':10s} {'prev':>5s} {'AUROC':>6s} {'medP+':>6s} {'medP-':>6s} | {'rec@0.5':>7s} {'spec@0.5':>8s} | "
      f"{'rec@spec.9':>10s} {'rec@spec.8':>10s} | {'spec@rec.8':>10s} {'spec@rec.9':>10s} | {'thr@rec.8':>9s}")
for i, c in enumerate(classes):
    y, s = Y[:, i], P[:, i]
    fpr, tpr, thr = roc_curve(y, s)
    spec = 1 - fpr
    r05, s05 = ((s >= 0.5) & (y == 1)).sum() / y.sum(), ((s < 0.5) & (y == 0)).sum() / (y == 0).sum()
    rec_at = lambda sp: tpr[spec >= sp].max()  # noqa: E731
    spec_at = lambda r: spec[tpr >= r].max()  # noqa: E731
    thr80 = thr[np.argmax(tpr >= 0.8)]
    print(f"{c:10s} {y.mean():5.2f} {roc_auc_score(y, s):6.3f} {np.median(s[y == 1]):6.3f} {np.median(s[y == 0]):6.3f} | "
          f"{r05:7.3f} {s05:8.3f} | {rec_at(0.9):10.3f} {rec_at(0.8):10.3f} | {spec_at(0.8):10.3f} {spec_at(0.9):10.3f} | "
          f"{thr80:9.3f}")
