"""Run process_scan in debug mode on a few scans and save raw/mask/window panels."""
import sys
import time

import numpy as np
import pandas as pd

from preprocess import process_scan
from utils import viz
from utils.config import load_config, p

cfg = load_config()
m = pd.read_csv(p(cfg, "manifest"))
m = m[m.usable]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
picks = pd.concat([m[m.axcodes == "LAS"].sample(n, random_state=3), m[~m.affine_consistent].sample(2, random_state=3)])
for _, r in picks.iterrows():
    t = time.time()
    s, d = process_scan(r.to_dict(), cfg, None, debug=True)
    key = s["key"]
    print(key, r.axcodes, f"{time.time() - t:.1f}s", {k: s[k] for k in ["n_raw", "n_kept", "mask_area_mean_cm2",
                                                                       "frac_50_80", "crop_mm"]})
    for z in [1, len(d["keep"]) // 2, len(d["keep"]) - 2]:
        viz.plot_preprocessing_check(d["hu"][z], d["gen"][z], d["out"][z].astype(np.float32) / 255,
                                     f"outputs/phase0/debug_pre/{key}_{z}.png", title=f"{key} z={d['keep'][z]}")
