"""Flag non-axial stacks (sagittal/coronal reformats) by left-right mirror symmetry of mid-level slices."""
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import load_config, p

cfg = load_config()
cache = p(cfg, "slice_cache")
out_dir = p(cfg, "outputs") / "orientation"


def sym(key):
    arr = np.load(cache / f"{key}.npy", mmap_mode="r")
    n = len(arr)
    vals = []
    for f in (0.45, 0.6, 0.75):
        s = arr[int(f * (n - 1)), 0].astype(np.float32)
        best = -1.0
        # allow a small horizontal shift of the midline
        for dx in range(-12, 13, 3):
            a = np.roll(s, dx, axis=1)
            b = a[:, ::-1]
            m = (a > 0) | (b > 0)
            best = max(best, float(np.corrcoef(a[m], b[m])[0, 1]))
        vals.append(best)
    return float(np.median(vals))


if __name__ == "__main__":
    o = pd.read_csv(out_dir / "orientation.csv")
    with ProcessPoolExecutor(24) as ex:
        o["lr_symmetry"] = list(ex.map(sym, o["key"], chunksize=32))
    o.to_csv(out_dir / "orientation.csv", index=False)
    print(o["lr_symmetry"].describe())
    low = o.sort_values("lr_symmetry").head(60)
    print(low[["key", "axcodes", "orient_status", "lr_symmetry"]].to_string())
    for page in range(3):
        sub = low.iloc[page * 20:(page + 1) * 20]
        fig, axes = plt.subplots(4, 5, figsize=(13, 11))
        for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
            arr = np.load(cache / f"{r['key']}.npy", mmap_mode="r")
            ax.imshow(arr[int(0.6 * (len(arr) - 1)), 0], cmap="gray")
            ax.set_title(f"{r['key']} {r['axcodes']} s={r['lr_symmetry']:.2f}", fontsize=7)
            ax.axis("off")
        fig.savefig(out_dir / f"lowsym_{page}.png", dpi=70, bbox_inches="tight")
        plt.close(fig)
