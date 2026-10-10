"""Dump raw intensity stats and mid-slice images for a few scans (orientation exploration)."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

m = pd.read_csv("outputs/scan_manifest.csv")
m = m[m.usable]
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
picks = pd.concat([m[m.axcodes == "LAS"].sample(4, random_state=1),
                   m[~m.affine_consistent].sample(4, random_state=1),
                   m[~m.axcodes.isin(["LAS", "LSP"])].head(2)])
fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
for r, (_, row) in enumerate(picks.iterrows()):
    img = nib.load(row.nifti_path)
    raw = np.asanyarray(img.dataobj)
    ax_s = int(row.slice_axis)
    vol = np.moveaxis(raw, ax_s, -1).astype(np.float32)
    print(row.nifti_path.split("/")[-1], row.axcodes, raw.shape, raw.dtype,
          "slope", img.dataobj.slope, "inter", img.dataobj.inter,
          "min/p1/p50/p99/max", np.percentile(vol, [0, 1, 50, 99, 100]).round(0))
    for k, frac in enumerate([0.3, 0.55, 0.8]):
        s = vol[..., int(frac * (vol.shape[-1] - 1))]
        axes[r, k].imshow(np.clip(s, 0, 80), cmap="gray")   # array displayed as stored (rows=axis0)
        axes[r, k].set_title(f"{row.axcodes} z={frac}", fontsize=7)
        axes[r, k].axis("off")
fig.savefig("outputs/phase0/peek_raw.png", dpi=80, bbox_inches="tight")
