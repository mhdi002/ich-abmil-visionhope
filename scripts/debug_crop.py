"""Show per-slice generous masks for a scan to debug off-center crops."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import process_scan
from utils.config import load_config, p

cfg = load_config()
m = pd.read_csv(p(cfg, "manifest"))
m["key"] = m.accession.astype(str) + "_" + m.ct_index.astype(str)
for key in sys.argv[1:]:
    row = m[m.key == key].iloc[0].to_dict()
    s, d = process_scan(row, cfg, None, debug=True)
    n = len(d["keep"])
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for i in range(n):
        ax = axes.ravel()[i]
        ax.imshow(np.clip(d["hu"][i], -100, 200), cmap="gray")
        ax.contour(d["gen"][i], levels=[0.5], colors="lime", linewidths=0.5)
        ax.contour(d["mask"][i], levels=[0.5], colors="red", linewidths=0.5)
        ax.set_title(f"z={d['keep'][i]}", fontsize=6)
    fig.savefig(f"outputs/phase0/debug_crop_{key}.png", dpi=70, bbox_inches="tight")
    print(key, s["crop_center"], s["crop_mm"], s["orig_hw"])
