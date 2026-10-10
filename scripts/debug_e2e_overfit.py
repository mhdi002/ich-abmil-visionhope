"""Isolate the end-to-end overfit failure: same 30 scans, vary backbone LR (0 = frozen encoder in the e2e path)."""
import copy
import sys

import numpy as np

from data.dataset import load_table
from train import train_stage_b
from utils.config import load_config, seed_everything

cfg = load_config(None)
seed_everything(cfg["seed"])
classes = cfg["classes"]
df = load_table(cfg)
tr = df[df["split"] == "train"]
rng = np.random.default_rng(cfg["seed"])
picks = []
for c in classes:
    picks += rng.choice(tr[tr[c] == 1]["key"].values, 4, replace=False).tolist()
picks += rng.choice(tr[tr[classes].sum(1) == 0]["key"].values, 30 - len(picks), replace=False).tolist()
sub = tr[tr["key"].isin(picks)].copy()

for lr_b in [float(v) for v in sys.argv[1:]]:
    c = copy.deepcopy(cfg)
    c["model"]["dropout"] = 0.0
    c["train"]["balanced_sampler"] = False
    c["train"]["ema_decay"] = 0.0
    c["train"]["num_workers"] = 4
    c["augment"].update(rotate_deg=0, scale=[1.0, 1.0], translate=0.0, contrast=0.0, brightness=0.0, noise_std=0.0)
    c["train"]["stage_b"].update(epochs=int(__import__("os").environ.get("EPOCHS", 12)), bags_per_step=1, grad_accum=2, lr_backbone=lr_b, lr_head=1e-3,
                                 weight_decay=0.0, warmup_epochs=0, patience=1000)
    t, _ = train_stage_b(c, f"debug_e2e_lr{lr_b:g}", train_df=sub, val_df=sub, eval_train_df=sub)
    print(f"RESULT lr_backbone={lr_b:g} train AUC per epoch:", [round(v, 3) for v in t.history["train_auc"]], flush=True)
