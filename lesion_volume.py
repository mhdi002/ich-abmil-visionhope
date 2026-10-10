"""Hyperdense lesion volume without segmentation labels (model 4 prototype).

Acute blood is 50-90 HU, brain 25-45 HU. Inside the intracranial mask of every cranial slice (posterior fossa
included, no end trim): median-filtered HU in [blood_hu], minus a thin band along the skull (partial-volume bone
edge mimics blood) and minus calcifications (pineal / choroid plexus: > calc_hu cores away from the skull). 3D
connected components below min_component_cm3 are dropped. Per scan: total volume, largest lesion, and the
extra-axial share (components touching the bone band = EDH/SDH-like) vs intra-axial (contusion / IVH-like).
No voxel labels exist, so the volumes are validated against scan-level proxies (eval stage): Marshall VI
(lesion > 25 cm3), Has_Any ICH, the 5 class labels, and lesion side.

  python lesion_volume.py --stage run --split val [--limit 200]
  python lesion_volume.py --stage eval --split val
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from preprocess import _disk, load_axial_hu, scan_masks
from utils.config import load_config, p, parse_overrides

DEFAULTS = dict(blood_hu=[50, 95], blood_delta_hu=20, brain_ref_hu=[15, 60], median_px=5, open_px=1,
                bone_band_mm=1.5, calc_hu=120, calc_margin_mm=2.0, calc_min_bone_dist_mm=4.0,
                min_component_cm3=0.2, extra_axial_touch_frac=0.15)


def lesion_volumes(row, cfg, lc, return_masks=False):
    hu, (row_mm, col_mm, slice_mm) = load_axial_hu(row)
    px = float(np.sqrt(row_mm * col_mm))
    sm = scan_masks(hu, px, row_mm, col_mm, cfg)
    keep = sm["keep"]
    if len(keep) == 0:
        return dict(ok=False)
    bone_hu = cfg["preprocess"]["bone_hu"]
    lo_hu, hi_hu = lc["blood_hu"]
    cand = np.zeros((len(keep),) + hu.shape[1:], bool)
    band = np.zeros_like(cand)
    for i, z in enumerate(keep):
        m = sm["masks"][z]
        if not m.any():
            continue
        h = ndi.median_filter(hu[z], size=lc["median_px"])
        dist_bone = ndi.distance_transform_edt(~(hu[z] > bone_hu)) * px
        near = dist_bone <= lc["bone_band_mm"]
        calc = (h > lc["calc_hu"]) & m & (dist_bone > lc["calc_min_bone_dist_mm"])
        if calc.any():
            calc = ndi.binary_dilation(calc, structure=_disk(lc["calc_margin_mm"] / px))
        # relative threshold: blood must be blood_delta_hu above this slice's brain (noise, scanner offset and
        # beam hardening under the vertex raise normal cortex above a fixed 50 HU)
        ref = h[m & (h > lc["brain_ref_hu"][0]) & (h < lc["brain_ref_hu"][1])]
        thr = max(lo_hu, float(np.median(ref)) + lc["blood_delta_hu"]) if ref.size > 100 else lo_hu
        c = m & (h >= thr) & (h <= hi_hu) & ~near & ~calc
        if lc["open_px"]:
            c = ndi.binary_opening(c, structure=_disk(lc["open_px"]))
        cand[i] = c
        band[i] = m & (dist_bone <= lc["bone_band_mm"] + 2.0)          # just inside the inner table
    vox_cm3 = row_mm * col_mm * slice_mm / 1000.0
    lab, n = ndi.label(cand, structure=np.ones((3, 3, 3), bool))
    comps = []
    if n:
        sizes = ndi.sum(cand, lab, index=np.arange(1, n + 1)) * vox_cm3
        touch = ndi.sum(band, lab, index=np.arange(1, n + 1)) / np.maximum(ndi.sum(cand, lab, index=np.arange(1, n + 1)), 1)
        boxes = ndi.find_objects(lab)
        for c in np.flatnonzero(sizes >= lc["min_component_cm3"]):
            zz, yy, xx = np.nonzero(lab[boxes[c]] == c + 1)
            zz, xx = zz + boxes[c][0].start, xx + boxes[c][2].start
            comps.append(dict(vol=float(sizes[c]), extra_axial=bool(touch[c] >= lc["extra_axial_touch_frac"]),
                              x_mean=float(xx.mean() / hu.shape[2]), n_slices=int(len(np.unique(zz)))))
    vols = [c["vol"] for c in comps]
    out = dict(ok=True, n_slices=int(len(keep)), vox_cm3=vox_cm3, n_lesions=len(comps),
               total_cm3=float(sum(vols)), largest_cm3=float(max(vols) if vols else 0.0),
               extra_axial_cm3=float(sum(c["vol"] for c in comps if c["extra_axial"])),
               intra_axial_cm3=float(sum(c["vol"] for c in comps if not c["extra_axial"])),
               # display convention: image left = patient right
               patient_right_cm3=float(sum(c["vol"] for c in comps if c["x_mean"] < 0.5)),
               patient_left_cm3=float(sum(c["vol"] for c in comps if c["x_mean"] >= 0.5)))
    if return_masks:
        keep_lab = np.isin(lab, [i + 1 for i, s in enumerate(ndi.sum(cand, lab, index=np.arange(1, n + 1)) * vox_cm3)
                                 if s >= lc["min_component_cm3"]]) if n else cand
        return out, dict(hu=hu[keep], lesion=keep_lab, masks=sm["masks"][keep], keep=keep)
    return out


def _worker(args):
    row, cfg, lc = args
    key = f"{int(row['accession'])}_{int(row['ct_index'])}"
    try:
        return dict(key=key, **lesion_volumes(row, cfg, lc))
    except Exception as e:  # noqa: BLE001
        return dict(key=key, ok=False, reason=repr(e)[:200])


def stage_run(cfg, lc, split, limit, workers):
    from data.dataset import load_table
    df = load_table(cfg)
    if split != "all":
        df = df[df["split"] == split]
    if limit:
        # keep Marshall VI over-represented so the 25 cm3 check has positives
        vi = df[df["Marshall CT score"] == 6]
        df = pd.concat([vi.sample(min(len(vi), limit // 4), random_state=0),
                        df[df["Marshall CT score"] != 6].sample(limit - min(len(vi), limit // 4), random_state=0)])
    out = p(cfg, "outputs") / "lesion_volume"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_worker, [(r, cfg, lc) for r in df.to_dict("records")], chunksize=4), 1):
            rows.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(df)}", flush=True)
    res = pd.DataFrame(rows)
    f = out / f"volumes_{split}{'_n' + str(limit) if limit else ''}.csv"
    res.to_csv(f, index=False)
    print("written", f, "ok", int(res["ok"].sum()), "/", len(res))
    return f


def stage_eval(cfg, split, limit):
    from sklearn.metrics import roc_auc_score
    from data.dataset import load_table
    f = p(cfg, "outputs") / "lesion_volume" / f"volumes_{split}{'_n' + str(limit) if limit else ''}.csv"
    v = pd.read_csv(f)
    d = load_table(cfg).merge(v[v["ok"] == True], on="key")  # noqa: E712
    rep = dict(n=len(d))
    vi = (d["Marshall CT score"] == 6).astype(int)
    for col in ("largest_cm3", "total_cm3"):
        rep[f"marshall_vi_auroc_{col}"] = float(roc_auc_score(vi, d[col]))
        pred = d[col] > 25
        rep[f"marshall_vi_at25_{col}"] = dict(sensitivity=float(pred[vi == 1].mean()),
                                              specificity=float((~pred)[vi == 0].mean()))
    rep["any_ich_auroc_total"] = float(roc_auc_score(d["Has_Any ICH"], d["total_cm3"]))
    rep["class_auroc_total"] = {c: float(roc_auc_score(d[c], d["total_cm3"])) for c in cfg["classes"]}
    rep["edh_sdh_auroc_extra_axial"] = {c: float(roc_auc_score(d[c], d["extra_axial_cm3"])) for c in ("EDH", "SDH")}
    rep["contusion_auroc_intra_axial"] = float(roc_auc_score(d["Contusion"], d["intra_axial_cm3"]))
    rep["median_largest_by_marshall"] = {int(k): float(g["largest_cm3"].median())
                                         for k, g in d.groupby("Marshall CT score")}
    rep["median_total_no_ich"] = float(d.loc[d["Has_Any ICH"] == 0, "total_cm3"].median())
    rep["median_total_ich"] = float(d.loc[d["Has_Any ICH"] == 1, "total_cm3"].median())
    # laterality agreement for unilateral SDH (radiologist side vs larger-volume side)
    uni_r = (d["SDH side_Right"] == 1) & (d["SDH side_Left"] == 0) & (d["SDH side_Bilateral"] == 0)
    uni_l = (d["SDH side_Left"] == 1) & (d["SDH side_Right"] == 0) & (d["SDH side_Bilateral"] == 0)
    side = np.where(d["patient_right_cm3"] > d["patient_left_cm3"], "R", "L")
    n_side = int(uni_r.sum() + uni_l.sum())
    rep["sdh_side_agreement"] = dict(n=n_side, agreement=float(((side == "R") & uni_r).sum() + ((side == "L") & uni_l).sum())
                                     / max(n_side, 1))
    print(json.dumps(rep, indent=1))
    (f.with_suffix(".eval.json")).write_text(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "eval"], required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(None, parse_overrides(args.set))
    lc = {**DEFAULTS, **(cfg.get("lesion_volume") or {})}
    if args.stage == "run":
        stage_run(cfg, lc, args.split, args.limit, args.workers)
    else:
        stage_eval(cfg, args.split, args.limit)


if __name__ == "__main__":
    main()
