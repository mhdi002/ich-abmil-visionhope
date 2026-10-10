"""Phase 0 gate: QC montages, orientation/laterality sanity, cheap baselines, 30-scan overfit tests.

  python phase0.py --step qc          # positive-example montages per class + laterality / AP checks
  python phase0.py --step baselines   # frozen ImageNet features (mean/max -> LR), HU features -> GBM/LR
  python phase0.py --step overfit     # 30-scan overfit: ABMIL head on frozen features, and end-to-end
"""
import argparse
import copy
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sstats  # noqa: E402

from data.dataset import load_table  # noqa: E402
from models.baselines import fit_per_class, gbm, logreg, pooled_features  # noqa: E402
from utils.config import load_config, p, parse_overrides, seed_everything  # noqa: E402

SUB_LO, SUB_HI = int(round((50 + 20) / 200 * 255)), int(round((90 + 20) / 200 * 255))  # 50-90 HU in subdural window


def report_path(cfg):
    d = p(cfg, "outputs") / "phase0"
    d.mkdir(parents=True, exist_ok=True)
    return d / "phase0_report.json"


def update_report(cfg, section, value):
    f = report_path(cfg)
    r = json.loads(f.read_text()) if f.exists() else {}
    r[section] = value
    f.write_text(json.dumps(r, indent=2, default=float))


def hyperdense_maps(arr, edge_px=6):
    """Per-slice 50-90 HU pixel map inside the tissue region, excluding a thin rim at the skull edge."""
    from scipy import ndimage as ndi
    sub = arr[:, 1]
    tissue = (sub > 0) & (sub < 230)
    hd = (sub >= SUB_LO) & (sub <= SUB_HI)
    out = np.zeros_like(hd)
    for i in range(len(arr)):
        inner = ndi.binary_erosion(tissue[i], iterations=edge_px)
        out[i] = hd[i] & inner
    return out


# ---------------------------------------------------------------- QC

def step_qc(cfg, per_class=20):
    seed_everything(cfg["seed"])
    classes = cfg["classes"]
    df = load_table(cfg, require_split=False)
    cache = p(cfg, "slice_cache")
    out = p(cfg, "plots") / "dataset" / "qc_positives"
    out.mkdir(parents=True, exist_ok=True)
    for c in classes:
        pos = df[df[c] == 1].sample(per_class, random_state=cfg["seed"])
        fig, axes = plt.subplots(4, 5, figsize=(15, 12.5))
        for ax, (_, r) in zip(axes.ravel(), pos.iterrows()):
            arr = np.load(cache / f"{r['key']}.npy")
            hd = hyperdense_maps(arr).reshape(len(arr), -1).sum(1)
            z = int(np.argmax(hd))
            ax.imshow(arr[z, 0], cmap="gray")
            others = ",".join(o for o in classes if r[o] == 1 and o != c)
            ax.set_title(f"{r['key']} z={z}/{len(arr)} +{others}", fontsize=7)
            ax.axis("off")
        fig.suptitle(f"{c}: 20 positive scans, slice with most 50-90 HU voxels (brain window, anterior up)")
        fig.savefig(out / f"qc_{c}.png", dpi=75, bbox_inches="tight")
        plt.close(fig)
    update_report(cfg, "orientation_checks", orientation_checks(cfg, df))
    print("QC montages ->", out)


def orientation_checks(cfg, df, max_scans=600):
    """Laterality (left/right) and anterior/posterior agreement of hyperdense voxels with labels."""
    cache = p(cfg, "slice_cache")
    res = {}

    def asym(keys, axis):
        vals = []
        for k in keys[:max_scans]:
            arr = np.load(cache / f"{k}.npy")
            n = len(arr)
            # mid-level slices only: skull-base partial volume dominates 50-90 HU counts posteriorly
            arr = arr[int(0.35 * n):max(int(0.85 * n), int(0.35 * n) + 1)]
            hd = hyperdense_maps(arr).sum(0).astype(np.float64)
            half = hd.shape[axis] // 2
            a = hd.take(range(half), axis=axis).sum()
            b = hd.take(range(half, hd.shape[axis]), axis=axis).sum()
            vals.append((a - b) / max(a + b, 1))
        return np.array(vals)

    for lesion in ("SDH", "EDH", "Contusion"):
        rcol, lcol, bcol = f"{lesion} side_Right", f"{lesion} side_Left", f"{lesion} side_Bilateral"
        if rcol not in df:
            continue
        right = df[(df[rcol] == 1) & (df[lcol] == 0) & (df.get(bcol, 0) == 0)]["key"].tolist()
        left = df[(df[lcol] == 1) & (df[rcol] == 0) & (df.get(bcol, 0) == 0)]["key"].tolist()
        a_r, a_l = asym(right, 1), asym(left, 1)   # (image-left - image-right) / total
        t = sstats.mannwhitneyu(a_r, a_l) if len(a_r) and len(a_l) else None
        res[f"{lesion}_laterality"] = dict(
            n_right=len(a_r), n_left=len(a_l),
            mean_imgleft_minus_imgright_for_right_labels=float(a_r.mean()) if len(a_r) else None,
            mean_imgleft_minus_imgright_for_left_labels=float(a_l.mean()) if len(a_l) else None,
            frac_right_labels_on_image_left=float((a_r > 0).mean()) if len(a_r) else None,
            frac_left_labels_on_image_right=float((a_l < 0).mean()) if len(a_l) else None,
            mannwhitney_p=float(t.pvalue) if t else None,
            implied_convention=("radiological (patient right on image left)" if len(a_r) and a_r.mean() > a_l.mean()
                                else "neurological (patient right on image right)"))
    fr = df[(df["Contusion site_Frontal"] == 1) & (df["Contusion site_Occipital"] == 0)]["key"].tolist()
    oc = df[(df["Contusion site_Occipital"] == 1) & (df["Contusion site_Frontal"] == 0)]["key"].tolist()
    a_f, a_o = asym(fr, 0), asym(oc, 0)   # (top - bottom) / total
    res["contusion_anterior_posterior"] = dict(
        n_frontal=len(a_f), n_occipital=len(a_o),
        mean_top_minus_bottom_frontal=float(a_f.mean()), mean_top_minus_bottom_occipital=float(a_o.mean()),
        frac_frontal_in_top_half=float((a_f > 0).mean()), frac_occipital_in_bottom_half=float((a_o < 0).mean()),
        mannwhitney_p=float(sstats.mannwhitneyu(a_f, a_o).pvalue))
    print(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- baselines

def _auc_table(y, prob, classes):
    from sklearn.metrics import average_precision_score, roc_auc_score
    t = {c: dict(auroc=float(roc_auc_score(y[:, i], prob[:, i])), auprc=float(average_precision_score(y[:, i], prob[:, i])))
         for i, c in enumerate(classes)}
    t["macro"] = dict(auroc=float(np.mean([t[c]["auroc"] for c in classes])),
                      auprc=float(np.mean([t[c]["auprc"] for c in classes])))
    return t


def step_baselines(cfg, encoders=("convnext_tiny", "visionhope_tiny")):
    seed_everything(cfg["seed"])
    classes = cfg["classes"]
    df = load_table(cfg)
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    Ytr, Yva = tr[classes].values, va[classes].values
    results = {}

    for enc in encoders:
        fdir = p(cfg, "feature_cache") / enc / "clean"
        if not fdir.exists():
            print(f"skip {enc}: no cached features at {fdir}")
            continue
        ftr = [np.load(fdir / f"{k}.npy").astype(np.float32) for k in tr["key"]]
        fva = [np.load(fdir / f"{k}.npy").astype(np.float32) for k in va["key"]]
        for how in ("mean", "max"):
            Xtr, Xva = pooled_features(ftr, how), pooled_features(fva, how)
            pr = fit_per_class(logreg, Xtr, Ytr, {"train": Xtr, "val": Xva})
            results[f"{enc}_{how}_logreg"] = dict(train=_auc_table(Ytr, pr["train"], classes),
                                                  val=_auc_table(Yva, pr["val"], classes))
            print(enc, how, "val macro AUROC", round(results[f"{enc}_{how}_logreg"]["val"]["macro"]["auroc"], 4),
                  {c: round(results[f"{enc}_{how}_logreg"]["val"][c]["auroc"], 3) for c in classes}, flush=True)

    hcols = [c for c in df.columns if c.startswith("hist_")] + [
        "hu_mean", "hu_std", "hu_p5", "hu_p25", "hu_p50", "hu_p75", "hu_p95", "frac_50_80", "frac_60_90",
        "vol_50_80_cm3", "slice_max_frac_50_80", "intracranial_vol_cm3", "mask_area_mean_cm2", "n_kept"]
    Xtr, Xva = tr[hcols].values, va[hcols].values
    for name, fn in (("hu_gbm", gbm), ("hu_logreg", logreg)):
        pr = fit_per_class(fn, Xtr, Ytr, {"train": Xtr, "val": Xva})
        results[name] = dict(train=_auc_table(Ytr, pr["train"], classes), val=_auc_table(Yva, pr["val"], classes))
        print(name, "val macro AUROC", round(results[name]["val"]["macro"]["auroc"], 4),
              {c: round(results[name]["val"][c]["auroc"], 3) for c in classes}, flush=True)
    update_report(cfg, "baselines", results)
    return results


# ---------------------------------------------------------------- overfit

def step_overfit(cfg, n=30, e2e=True):
    from train import extract_features, train_stage_a, train_stage_b
    seed_everything(cfg["seed"])
    classes = cfg["classes"]
    df = load_table(cfg)
    tr = df[df["split"] == "train"]
    # Pick scans so each class has positives and negatives among the 30.
    picks = []
    rng = np.random.default_rng(cfg["seed"])
    for c in classes:
        picks += rng.choice(tr[tr[c] == 1]["key"].values, 4, replace=False).tolist()
    picks += rng.choice(tr[tr[classes].sum(1) == 0]["key"].values, n - len(picks), replace=False).tolist()
    sub = tr[tr["key"].isin(picks)].copy()
    res = {"n_scans": int(len(sub)), "positives": {c: int(sub[c].sum()) for c in classes}}

    c = copy.deepcopy(cfg)
    c["model"]["dropout"] = 0.0
    c["train"]["balanced_sampler"] = False
    c["train"]["ema_decay"] = 0.0
    c["train"]["stage_a"].update(epochs=150, batch_size=8, lr_head=1e-3, weight_decay=0.0, warmup_epochs=1, patience=1000)
    tracker, _ = train_stage_a(c, "phase0_overfit_head", train_df=sub, val_df=sub, quiet_plots=True)
    res["head_on_frozen_features"] = dict(final_train_auc=tracker.history["train_auc"][-1],
                                          max_train_auc=max(tracker.history["train_auc"]),
                                          final_train_loss=tracker.history["train_loss"][-1])
    print("overfit head:", res["head_on_frozen_features"], flush=True)

    if e2e:
        c = copy.deepcopy(cfg)
        c["model"]["dropout"] = 0.0
        c["train"]["balanced_sampler"] = False
        c["train"]["ema_decay"] = 0.0
        c["train"]["num_workers"] = 4
        c["augment"].update(rotate_deg=0, scale=[1.0, 1.0], translate=0.0, contrast=0.0, brightness=0.0, noise_std=0.0)
        c["train"]["stage_b"].update(epochs=40, bags_per_step=1, grad_accum=2, lr_backbone=1e-5, lr_head=1e-3, weight_decay=0.0,
                                     warmup_epochs=0, patience=1000)
        tracker, _ = train_stage_b(c, "phase0_overfit_e2e", train_df=sub, val_df=sub, eval_train_df=sub)
        res["end_to_end"] = dict(final_train_auc=tracker.history["train_auc"][-1],
                                 max_train_auc=max(tracker.history["train_auc"]),
                                 final_train_loss=tracker.history["train_loss"][-1])
        print("overfit e2e:", res["end_to_end"], flush=True)
    res["gate_pass"] = bool(res["head_on_frozen_features"]["max_train_auc"] > 0.97 and
                            (not e2e or res["end_to_end"]["max_train_auc"] > 0.97))
    update_report(cfg, "overfit", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["qc", "baselines", "overfit"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-e2e", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    if args.step == "qc":
        step_qc(cfg)
    elif args.step == "baselines":
        step_baselines(cfg)
    else:
        step_overfit(cfg, e2e=not args.no_e2e)


if __name__ == "__main__":
    main()
