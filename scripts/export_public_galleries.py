"""Re-render attention + Grad-CAM galleries with anonymous labels (no scan keys) for publication.

Usage: python scripts/export_public_galleries.py [run] [out_dir]
Uses the cached val predictions and the F1 thresholds from eval_report.json of the run.
"""
import json
import sys
from pathlib import Path

import numpy as np

from data.dataset import load_table
from eval import galleries, load_run
from utils.config import load_config, p, seed_everything

run = sys.argv[1] if len(sys.argv) > 1 else "e2e_vh"
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/figures/eval/e2e_vh")
out_dir.mkdir(parents=True, exist_ok=True)
cur = load_config()
run_dir = p(cur, "outputs") / "runs" / run
cfg, model, ck = load_run(run_dir)
cfg["paths"] = {**cfg["paths"], **cur["paths"]}       # data locations are machine-specific
seed_everything(cfg["seed"])
classes = cfg["classes"]
rep = json.loads((run_dir / "eval_report.json").read_text())
thr = np.array([rep["thresholds"][c] for c in classes])
z = np.load(run_dir / "preds_val.npz", allow_pickle=True)
preds = dict(keys=list(z["keys"]), y=z["y"], prob=z["prob"], attn=list(z["attn"]))
df = load_table(cfg)
galleries(cfg, model, df[df["split"] == "val"], preds, thr, classes, out_dir, anonymize=True)
print("galleries ->", out_dir)
