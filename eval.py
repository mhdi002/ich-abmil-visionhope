"""Phase 5 evaluation.

  python eval.py --run abmil_vh                      # full evaluation of one run (train/val/test)
  python eval.py --ablation run1 run2 ...            # val-set ablation table across runs
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import FeatureBagDataset, SliceBagDataset, gpu_prepare, load_slices, load_table, pad_collate
from models.abmil import BagModel, build_head
from models.encoder import build_encoder
from models.losses import build_loss
from utils import viz
from utils.config import load_config, p, parse_overrides, seed_everything

STRATA_MIN = 30


# ---------------------------------------------------------------- loading / prediction

def load_run(run_dir):
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    encoder = build_encoder(cfg)
    if ck["stage"] == "A":
        head = build_head(cfg, ck["in_dim"])
        head.load_state_dict(ck["head"])
        model = BagModel(encoder, head)
    else:
        model = BagModel(encoder, build_head(cfg, encoder.out_dim))
        model.load_state_dict(ck["model"])
    return cfg, model.cuda().eval(), ck


def predict(cfg, model, ck, df, run_dir, split):
    cache = run_dir / f"preds_{split}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        res = dict(keys=list(z["keys"]), y=z["y"], prob=z["prob"], attn=list(z["attn"]))
        res.update({k: z[k] for k in ("task_y", "task_prob", "reg_y", "reg_pred", "aux_prob") if k in z.files})
        return res
    from train import eval_bags, eval_features, feature_dir
    crit = build_loss(cfg, df[cfg["classes"]].values).cuda()
    if ck["stage"] == "A":
        dl = DataLoader(FeatureBagDataset(df, cfg, feature_dir(cfg), ["clean"]), batch_size=64,
                        num_workers=8, collate_fn=pad_collate)
        res = eval_features(model.head, dl, crit, cfg)
    else:
        dl = DataLoader(SliceBagDataset(df, cfg), batch_size=1, num_workers=8, collate_fn=pad_collate)
        res = eval_bags(model, dl, crit, cfg)
    attn = np.empty(len(res["attn"]), dtype=object)
    attn[:] = res["attn"]
    extra = {k: res[k] for k in ("task_y", "task_prob", "reg_y", "reg_pred", "aux_prob") if res.get(k) is not None}
    np.savez(cache, keys=np.array(res["keys"]), y=res["y"], prob=res["prob"], attn=attn, **extra)
    return res


# ---------------------------------------------------------------- metrics

def recall_at_spec(y, s, spec=0.9):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


THRESH_KEYS = ("accuracy", "balanced_accuracy", "recall", "precision", "specificity")


def class_metrics(y, s, thr, spec):
    if not (0 < y.sum() < len(y)):
        return dict(auroc=np.nan, auprc=np.nan, f1=np.nan, recall_at_spec=np.nan, **{k: np.nan for k in THRESH_KEYS})
    (tm,), _ = viz.threshold_metrics(y[:, None], s[:, None], thr)
    return dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s),
                f1=f1_score(y, (s >= thr).astype(int), zero_division=0), recall_at_spec=recall_at_spec(y, s, spec),
                **{k: float(tm[k]) for k in THRESH_KEYS})


def all_metrics(Y, P, thr, classes, spec):
    out = {c: class_metrics(Y[:, i], P[:, i], thr[i], spec) for i, c in enumerate(classes)}
    out["macro"] = {m: float(np.nanmean([out[c][m] for c in classes])) for m in out[classes[0]]}
    return out


def bootstrap_ci(Y, P, groups, thr, classes, spec, n_boot, seed):
    """Patient-clustered bootstrap (serial CTs of a patient are resampled together)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    samples = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        samples.append(all_metrics(Y[idx], P[idx], thr, classes, spec))
    ci = {}
    for c in classes + ["macro"]:
        ci[c] = {m: (float(np.nanpercentile([s[c][m] for s in samples], 2.5)),
                     float(np.nanpercentile([s[c][m] for s in samples], 97.5))) for m in samples[0][c]}
    return ci


def choose_thresholds(Y, P, mode="f1", target_recall=0.8):
    """Per-class thresholds chosen on val. mode: 'f1' (max F1) or 'recall' (highest threshold with recall >= target)."""
    if mode == "recall":
        thr = []
        for i in range(Y.shape[1]):
            pos = np.sort(P[Y[:, i] == 1, i])
            k = int(np.floor((1 - target_recall) * len(pos)))
            thr.append(float(pos[min(k, len(pos) - 1)]) if len(pos) else 0.5)
        return np.array(thr)
    thr = []
    for i in range(Y.shape[1]):
        cands = np.unique(np.quantile(P[:, i], np.linspace(0.01, 0.99, 197)))
        f1s = [f1_score(Y[:, i], (P[:, i] >= t).astype(int), zero_division=0) for t in cands]
        thr.append(float(cands[int(np.argmax(f1s))]))
    return np.array(thr)


def stratified(df, P, classes, col, bins=None):
    d = df.copy()
    if bins is not None:
        d[col + "_grp"] = bins(d[col])
        col = col + "_grp"
    rows = []
    for g, idx in d.groupby(col).groups.items():
        idx = d.index.get_indexer(idx)
        if len(idx) < STRATA_MIN:
            continue
        Y = d[classes].values[idx]
        row = dict(stratum=str(g), n=len(idx))
        aucs = []
        for i, c in enumerate(classes):
            if 0 < Y[:, i].sum() < len(Y) and Y[:, i].sum() >= 5:
                a = roc_auc_score(Y[:, i], P[idx, i])
                row[c] = round(float(a), 4)
                aucs.append(a)
            else:
                row[c] = None
            row[f"{c}_prev"] = round(float(Y[:, i].mean()), 3)
        row["macro_auroc"] = round(float(np.mean(aucs)), 4) if aucs else None
        rows.append(row)
    return rows


# ---------------------------------------------------------------- probes and orientation

def later_operated_probe(cfg, df):
    from models.baselines import logreg, pooled_features
    fdir = p(cfg, "feature_cache") / cfg["encoder"]["name"] / "clean"
    out = {}
    tr = df[df["split"] == "train"]
    Xtr = pooled_features([np.load(fdir / f"{k}.npy").astype(np.float32) for k in tr["key"]], "both")
    mdl = logreg().fit(Xtr, tr["later_operated"].astype(int))
    for s in ("val", "test"):
        d = df[df["split"] == s]
        X = pooled_features([np.load(fdir / f"{k}.npy").astype(np.float32) for k in d["key"]], "both")
        out[f"{s}_auroc"] = float(roc_auc_score(d["later_operated"].astype(int), mdl.predict_proba(X)[:, 1]))
        # Same probe restricted to label-matched pairs: does it still separate when hemorrhage labels are equal?
        neg = d[d[cfg["classes"]].sum(1) == 0]
        if neg["later_operated"].nunique() == 2:
            Xn = pooled_features([np.load(fdir / f"{k}.npy").astype(np.float32) for k in neg["key"]], "both")
            out[f"{s}_auroc_label_negative_only"] = float(roc_auc_score(neg["later_operated"].astype(int),
                                                                        mdl.predict_proba(Xn)[:, 1]))
    return out


def attention_laterality(cfg, df, preds, classes):
    """For unilateral SDH/EDH true positives: does the top-attention slice show more 50-90 HU on the
    image side implied by the dataset convention?"""
    from phase0 import hyperdense_maps
    cache = p(cfg, "slice_cache")
    idx = {k: i for i, k in enumerate(preds["keys"])}
    out = {}
    for lesion in ("SDH", "EDH"):
        c = classes.index(lesion)
        res = []
        for side, sign in (("Right", 1), ("Left", -1)):
            other = "Left" if side == "Right" else "Right"
            d = df[(df[f"{lesion} side_{side}"] == 1) & (df[f"{lesion} side_{other}"] == 0) &
                   (df[f"{lesion} side_Bilateral"] == 0) & (df[lesion] == 1)]
            for k in d["key"]:
                i = idx.get(k)
                if i is None or preds["prob"][i, c] < 0.5:
                    continue
                arr = load_slices(cache, k)
                z = int(np.argmax(preds["attn"][i][c]))
                hd = hyperdense_maps(arr[z:z + 1])[0].astype(float)
                half = hd.shape[1] // 2
                a = (hd[:, :half].sum() - hd[:, half:].sum()) / max(hd.sum(), 1)
                res.append(sign * a)
        res = np.array(res)
        out[lesion] = dict(n=len(res), frac_expected_side_radiological=float((res > 0).mean()) if len(res) else None,
                           mean_signed_asymmetry=float(res.mean()) if len(res) else None)
    return out


# ---------------------------------------------------------------- Grad-CAM / galleries

def gradcam_slices(model, x_bag, cls, top_idx):
    """Grad-CAM on selected slices for bag-level logit ``cls``. x_bag: [N,3,H,W] prepared."""
    enc = model.encoder
    stage = enc.last_stage()
    acts = {}

    def hook(_, __, out):
        out.retain_grad()
        acts["a"] = out
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([enc(x_bag[i:i + 64]).float() for i in range(0, len(x_bag), 64)])
    handle = stage.register_forward_hook(hook)
    try:
        sel = x_bag[top_idx].clone().requires_grad_(False)
        with torch.enable_grad():
            hs = enc(sel).float()
            h2 = h.clone()
            h2[top_idx] = hs
            out = model.head(h2[None], torch.ones(1, len(h2), dtype=torch.bool, device=h2.device))
            out["logits"][0, cls].backward()
        a, g = acts["a"], acts["a"].grad
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(1)).detach()
        cam = torch.nn.functional.interpolate(cam[:, None], size=x_bag.shape[-2:], mode="bilinear")[:, 0]
        # display only: zero the map outside the masked head (cached pixels are 0 there in every window)
        head = (x_bag[top_idx].amax(1, keepdim=True) > 0).float()
        head = torch.nn.functional.max_pool2d(head, 7, stride=1, padding=3)[:, 0]
        cam = cam * head
        cam = cam / cam.flatten(1).max(1).values.clamp(min=1e-8)[:, None, None]
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    return cam.cpu().numpy()


def galleries(cfg, model, df, preds, thr, classes, out_dir, k_each=3, topk=4, anonymize=False):
    """anonymize: label examples '<class> <TP|FP|FN> #k' instead of the scan key (keys contain accession numbers)."""
    cache = p(cfg, "slice_cache")
    idx = {key: i for i, key in enumerate(preds["keys"])}
    rng = np.random.default_rng(cfg["seed"])
    for c, name in enumerate(classes):
        y = preds["y"][:, c]
        s = preds["prob"][:, c]
        groups = {"TP": np.flatnonzero((y == 1) & (s >= thr[c])), "FP": np.flatnonzero((y == 0) & (s >= thr[c])),
                  "FN": np.flatnonzero((y == 1) & (s < thr[c]))}
        items = {}
        for g, ids in groups.items():
            if len(ids) == 0:
                continue
            ids = ids[np.argsort(-s[ids])][:20] if g != "FN" else ids[np.argsort(s[ids])][:20]
            ids = rng.choice(ids, min(k_each, len(ids)), replace=False)
            items[g] = []
            for j, i in enumerate(ids, 1):
                key = preds["keys"][i]
                arr = load_slices(cache, key)
                a = preds["attn"][i][c]
                top = np.argsort(-a)[:topk]
                x = gpu_prepare(torch.from_numpy(arr).cuda(), cfg, train=False)
                try:
                    cam = gradcam_slices(model, x, c, torch.as_tensor(top, device="cuda"))
                except Exception as e:  # noqa: BLE001
                    print("grad-cam failed:", e)
                    cam = None
                items[g].append(dict(slices=x[top].cpu().numpy(), attn=a[top], prob=float(s[i]),
                                     name=f"#{j}" if anonymize else key, cam=cam))
        viz.plot_attention_gallery(items, name, out_dir / f"attention_gallery_{name}.png")


# ---------------------------------------------------------------- main evaluation

def evaluate_run(run, test=True, eval_overrides=None, cfg_overrides=None):
    cur = load_config(None, cfg_overrides)     # --set paths.* points at another cache / outputs dir
    run_dir = p(cur, "outputs") / "runs" / run
    cfg, model, ck = load_run(run_dir)
    cfg["eval"] = {**cfg.get("eval", {}), **cur["eval"], **(eval_overrides or {})}
    cfg["paths"] = {**cfg["paths"], **cur["paths"]}       # data locations are machine-specific
    seed_everything(cfg["seed"])
    classes = cfg["classes"]
    spec, n_boot = cfg["eval"]["spec_target"], cfg["eval"]["n_bootstrap"]
    df = load_table(cfg)
    tag = "" if cfg["eval"].get("threshold_mode", "f1") == "f1" else f"_recall{int(100 * cfg['eval']['target_recall'])}"
    out_dir = p(cfg, "plots") / "eval" / f"{run}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(p(cfg, "tensorboard") / f"eval_{run}"))
    splits = ["train", "val"] + (["test"] if test else [])
    preds, dfs = {}, {}
    for s in splits:
        d = df[df["split"] == s].reset_index(drop=True)
        r = predict(cfg, model, ck, d, run_dir, s)
        d = d.set_index("key").loc[r["keys"]].reset_index()
        assert (d[classes].values == r["y"]).all()
        preds[s], dfs[s] = r, d
    ev = cfg["eval"]
    thr = choose_thresholds(preds["val"]["y"], preds["val"]["prob"], ev.get("threshold_mode", "f1"),
                            ev.get("target_recall", 0.8))
    report = dict(run=run, stage=ck["stage"], best_epoch=ck["epoch"], encoder=cfg["encoder"]["name"],
                  threshold_mode=ev.get("threshold_mode", "f1"), target_recall=ev.get("target_recall"),
                  pooling=cfg["model"]["pooling"], attention=cfg["model"]["attention"],
                  srnl=cfg["model"]["srnl_context"], loss=cfg["loss"]["name"],
                  thresholds={c: float(t) for c, t in zip(classes, thr)}, splits={})
    for s in splits:
        Y, P = preds[s]["y"], preds[s]["prob"]
        m = all_metrics(Y, P, thr, classes, spec)
        ci = bootstrap_ci(Y, P, dfs[s]["accession"].values, thr, classes, spec,
                          n_boot if s != "train" else max(n_boot // 5, 100), cfg["seed"])
        report["splits"][s] = dict(n=len(Y), metrics=m, ci95=ci)
        viz.save_full_eval_plots(Y, P, thr, classes, out_dir, tag=s, writer=writer)
        viz.plot_metric_bars_with_ci({k: {c: (m[c][k], ci[c][k][0], ci[c][k][1]) for c in classes + ["macro"]}
                                      for k in ("auroc", "auprc", "accuracy", "balanced_accuracy", "recall",
                                                "specificity")},
                                     classes, out_dir / f"{s}_metric_bars.png", tag=s)
    for s in [x for x in splits if x != "train"]:
        d, P = dfs[s], preds[s]["prob"]
        report.setdefault("stratified", {})[s] = dict(
            later_operated=stratified(d, P, classes, "later_operated"),
            ct_index=stratified(d, P, classes, "ct_index",
                                lambda v: pd.cut(v, [0, 1, 2, 100], labels=["1", "2", "3+"])),
            radiologist=stratified(d, P, classes, "Radiologists"),
            marshall=stratified(d, P, classes, "Marshall CT score"),
            chrono_holdout=stratified(d, P, classes, "chrono_holdout"))
    try:
        report["later_operated_probe"] = later_operated_probe(cfg, df)
    except FileNotFoundError:
        # no file path in the report: cache file names are scan keys (accession numbers)
        report["later_operated_probe"] = "skipped: frozen-feature cache not available for this encoder"
    report["attention_laterality_val"] = attention_laterality(cfg, dfs["val"], preds["val"], classes)
    galleries(cfg, model, dfs["val"], preds["val"], thr, classes, out_dir)
    (run_dir / f"eval_report{tag}.json").write_text(json.dumps(report, indent=1, default=float))
    print_summary(report, classes)
    writer.close()
    return report


def print_summary(report, classes):
    print(f"\n=== {report['run']} (stage {report['stage']}, best epoch {report['best_epoch']}) ===")
    hdr = f"{'split':6s} " + " ".join(f"{c:>17s}" for c in classes + ['macro'])
    for metric in ("auroc", "auprc", "accuracy", "balanced_accuracy", "recall", "precision", "specificity", "f1",
                   "recall_at_spec"):
        print(f"-- {metric} (95% CI)")
        print(hdr)
        for s, r in report["splits"].items():
            cells = [f"{r['metrics'][c][metric]:.3f} [{r['ci95'][c][metric][0]:.2f},{r['ci95'][c][metric][1]:.2f}]"
                     for c in classes + ["macro"]]
            print(f"{s:6s} " + " ".join(f"{x:>17s}" for x in cells))


def ablation_table(runs):
    cfg0 = load_config()
    rows = []
    for run in runs:
        run_dir = p(cfg0, "outputs") / "runs" / run
        f = run_dir / "eval_report.json"
        if not f.exists():
            evaluate_run(run, test=False)
        r = json.loads(f.read_text())
        v = r["splits"]["val"]["metrics"]
        t = r["splits"]["train"]["metrics"]
        row = dict(run=run, encoder=r["encoder"], pooling=r["pooling"], attention=r["attention"],
                   srnl=str(r["srnl"]), loss=r["loss"], train_auroc=t["macro"]["auroc"],
                   val_auroc=v["macro"]["auroc"], val_auprc=v["macro"]["auprc"])
        for c in cfg0["classes"]:
            row[f"val_{c}"] = v[c]["auroc"]
        rows.append(row)
    out = p(cfg0, "plots") / "eval" / "ablation_table.png"
    viz.plot_ablation_table(rows, out)
    pd.DataFrame(rows).to_csv(p(cfg0, "outputs") / "ablation_table.csv", index=False)
    print(pd.DataFrame(rows).to_string(float_format=lambda x: f"{x:.4f}"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--ablation", nargs="*", default=None)
    ap.add_argument("--threshold-mode", choices=["f1", "recall"], default=None)
    ap.add_argument("--target-recall", type=float, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    ov = {k: v for k, v in dict(threshold_mode=args.threshold_mode, target_recall=args.target_recall).items()
          if v is not None}
    if args.ablation:
        ablation_table(args.ablation)
    else:
        evaluate_run(args.run, test=not args.no_test, eval_overrides=ov, cfg_overrides=parse_overrides(args.set))


if __name__ == "__main__":
    main()
