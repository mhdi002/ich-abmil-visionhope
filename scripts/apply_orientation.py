"""Apply the orientation fixes already decided for the main cache to another slice cache (e.g. the cistern cache).

The main cache was fixed in place by `preprocess.py --stage orient` twice (orientation_pass1.csv, then
orientation.csv on the already-rotated cache); the total rotation per scan is the sum of both passes' confident
fixes. Handles .npy and compressed .npz caches.

  python scripts/apply_orientation.py --cache "<outputs_cistern>/cache/slices"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_config, p  # noqa: E402


def total_rotation(orient_dir):
    k = None
    for name in ("orientation_pass1.csv", "orientation.csv"):
        f = orient_dir / name
        if not f.exists():
            continue
        o = pd.read_csv(f)
        kk = pd.Series(np.where(o["orient_status"] == "fixed", o["orient_fix_k"], 0), index=o["key"])
        k = kk if k is None else k.add(kk, fill_value=0)
    return (k % 4).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    cfg = load_config()
    cache = Path(args.cache)
    marker = cache / ".orientation_applied"
    if marker.exists():
        print(f"orientation already applied to {cache}")
        return
    k = total_rotation(p(cfg, "outputs") / "orientation")
    todo = k[k != 0]
    n = 0
    for key, r in todo.items():
        npy, npz = cache / f"{key}.npy", cache / f"{key}.npz"
        if npy.exists():
            np.save(npy, np.ascontiguousarray(np.rot90(np.load(npy), int(r), axes=(-2, -1))))
        elif npz.exists():
            with np.load(npz) as z:
                x = z["x"]
            np.savez_compressed(npz, x=np.ascontiguousarray(np.rot90(x, int(r), axes=(-2, -1))))
        else:
            continue
        n += 1
    marker.write_text(f"rotated {n}\n")
    print(f"rotated {n} scans in {cache} ({len(todo)} with a non-zero total rotation)")


if __name__ == "__main__":
    main()
