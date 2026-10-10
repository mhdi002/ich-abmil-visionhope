"""QC of the slice trim: which end slices are dropped (skull base / vertex) on a random sample of scans.

  python scripts/trim_qc.py --n 40 --montages 24 [--set preprocess.trim_base_bone_frac=0.05]

Per scan, a montage of every slice in the cranial range (bone-inclusive window so petrous bone, orbits and
the sphenoid are visible): green = kept, red = dropped; titles give intracranial area / max and deep bone /
max area. Prints how many slices each rule removes and the resulting bag lengths. Images contain scan keys:
local QC only, written under outputs/plots/trim_qc/.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import process_scan  # noqa: E402
from utils.config import load_config, p, parse_overrides  # noqa: E402


def run_one(args):
    row, cfg = args
    try:
        s, dbg = process_scan(row, cfg, None, debug=True)
    except Exception as e:  # noqa: BLE001
        return dict(key=f"{int(row['accession'])}_{int(row['ct_index'])}", ok=False, reason=repr(e)[:200]), None
    if not s.get("ok"):
        return s, None
    return s, dict(hu=dbg["hu_all"][dbg["keep_pre"]], keep_pre=dbg["keep_pre"], keep=dbg["keep"])


def montage(key, s, d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hu, pre, kept = d["hu"], d["keep_pre"], set(d["keep"].tolist())
    area, bone = np.asarray(s.get("trim_area", [np.nan] * len(pre))), np.asarray(s.get("trim_bone", [np.nan] * len(pre)))
    amax = np.nanmax(area) if len(area) else 1.0
    n = len(pre)
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (ax, z) in enumerate(zip(axes.ravel(), pre)):
        ax.imshow(np.clip((hu[i] + 100) / 1100.0, 0, 1), cmap="gray")       # -100..1000 HU: brain + bone
        ok = z in kept
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("lime" if ok else "red")
            sp.set_linewidth(3)
        ax.set_title(f"z{z} a={area[i] / amax:.2f} b={bone[i] / amax:.3f}", fontsize=7,
                     color="green" if ok else "red")
    fig.suptitle(f"{key}: kept {len(kept)}/{n} (green kept, red dropped; inferior first)", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--montages", type=int, default=24)
    ap.add_argument("--keys", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(None, parse_overrides(args.set))
    m = pd.read_csv(p(cfg, "manifest"))
    m = m[m["usable"]]
    m["key"] = m["accession"].astype(int).astype(str) + "_" + m["ct_index"].astype(int).astype(str)
    sub = m[m["key"].isin(args.keys)] if args.keys else m.sample(args.n, random_state=cfg["seed"])
    out_dir = p(cfg, "plots") / "trim_qc"
    rows = []
    with ProcessPoolExecutor(cfg["preprocess"]["workers"]) as ex:
        for i, (s, d) in enumerate(ex.map(run_one, [(r, cfg) for r in sub.to_dict("records")])):
            if d is None:
                print("failed", s.get("key"), s.get("reason"))
                continue
            base, vertex = s["anatomic_trim"]
            rows.append(dict(key=s["key"], pre=len(d["keep_pre"]), kept=len(d["keep"]), base=base, vertex=vertex))
            if i < args.montages:
                montage(s["key"], s, d, out_dir / f"{s['key']}.png")
    df = pd.DataFrame(rows)
    print(df.describe().round(2).to_string())
    print(f"median bag {df['pre'].median():.0f} -> {df['kept'].median():.0f}; "
          f"removed base {df['base'].sum()} vertex {df['vertex'].sum()} slices over {len(df)} scans")
    print("montages:", out_dir)


if __name__ == "__main__":
    main()
