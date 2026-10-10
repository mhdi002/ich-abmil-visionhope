"""Multi-task evaluation: task heads, midline-shift regression, and a Marshall CT score built from them.

  python scripts/eval_marshall.py --run e2e_vh_mt

Marshall rule (Marshall et al. 1991; category V = surgically evacuated is not predictable from a pre-op image and
is excluded, 10 cohort scans):
  VI  high/mixed density lesion > 25 cm3       <- head marshall_vi (the volume model can replace it later)
  IV  midline shift > 5 mm                      <- head mls_gt5
  III cisterns compressed or absent             <- head cist_compressed
  II  any visible intracranial pathology        <- head marshall_ge2
  I   otherwise
Thresholds: the 4 rule thresholds are tuned jointly on val (max quadratic kappa vs the radiologist score) and
applied unchanged to test; per-head max-F1 thresholds are reported for comparison. Also reported: the same rule
applied to the radiologist's own
component labels (how well the reported components reproduce the reported score = label-consistency ceiling).
Patient-clustered bootstrap CIs. Writes <run>/marshall_report.json and plots under plots/eval/<run>/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_table, task_names  # noqa: E402
from eval import all_metrics, bootstrap_ci, choose_thresholds  # noqa: E402
from utils.config import load_config, p, parse_overrides  # noqa: E402

CATS = [1, 2, 3, 4, 6]
RULE = [("marshall_vi", 6), ("mls_gt5", 4), ("cist_compressed", 3), ("marshall_ge2", 2)]


def marshall_rule(flags):
    """flags: {head: bool array} -> Marshall category per scan."""
    n = len(next(iter(flags.values())))
    out = np.ones(n, dtype=int)
    for head, cat in reversed(RULE):          # lowest priority first, higher categories overwrite
        out[flags[head]] = cat
    return out


def tune_rule_thresholds(P, idx, truth, start, rounds=4):
    """Coordinate ascent on val: the 4 rule thresholds jointly, maximizing quadratic-weighted kappa vs the
    radiologist score (per-head F1 thresholds ignore that the heads are combined in priority order)."""
    thr = dict(start)
    grid = {h: np.unique(np.quantile(P[:, idx[h]], np.linspace(0.02, 0.995, 120))) for h, _ in RULE}

    def score(t):
        pred = marshall_rule({h: P[:, idx[h]] >= t[h] for h, _ in RULE})
        return cohen_kappa_score(truth, pred, weights="quadratic")
    best = score(thr)
    for _ in range(rounds):
        changed = False
        for h, _ in RULE:
            for v in grid[h]:
                cand = {**thr, h: float(v)}
                s = score(cand)
                if s > best + 1e-6:
                    best, thr, changed = s, cand, True
        if not changed:
            break
    return thr, float(best)


def kappa_ci(y, yhat, groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    ks = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        ks.append(cohen_kappa_score(y[idx], yhat[idx], weights="quadratic"))
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))


def marshall_metrics(y, yhat, groups, n_boot, seed):
    cm = confusion_matrix(y, yhat, labels=CATS)
    rec = {f"cat{c}": float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(CATS)}
    prec = {f"cat{c}": float(cm[i, i] / max(cm[:, i].sum(), 1)) for i, c in enumerate(CATS)}
    return dict(n=int(len(y)), accuracy=float((y == yhat).mean()),
                kappa_quadratic=float(cohen_kappa_score(y, yhat, weights="quadratic")),
                kappa_quadratic_ci95=kappa_ci(y, yhat, groups, n_boot, seed),
                kappa_linear=float(cohen_kappa_score(y, yhat, weights="linear")),
                recall=rec, precision=prec, confusion=cm.tolist(), labels=CATS)


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = ["I", "II", "III", "IV", "VI"]
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted (rule over model heads)")
    ax.set_ylabel("radiologist Marshall")
    ax.set_title(title, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e2e_vh_mt")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cur = load_config(None, parse_overrides(args.set))
    run_dir = p(cur, "outputs") / "runs" / args.run
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg["paths"] = {**cfg["paths"], **cur["paths"]}
    tb, tr = task_names(cfg)
    n_boot, seed = cur["eval"]["n_bootstrap"], cfg["seed"]
    df = load_table(cfg).set_index("key")
    preds = {}
    for s in ("val", "test"):
        z = np.load(run_dir / f"preds_{s}.npz", allow_pickle=True)
        preds[s] = dict(keys=[str(k) for k in z["keys"]], task_y=z["task_y"], task_prob=z["task_prob"],
                        reg_y=z["reg_y"], reg_pred=z["reg_pred"])
    report = dict(run=args.run, tasks={}, regression={}, marshall={})

    # binary heads
    thr = choose_thresholds(preds["val"]["task_y"], preds["val"]["task_prob"], "f1")
    report["task_thresholds"] = {k: float(t) for k, t in zip(tb, thr)}
    for s in ("val", "test"):
        Y, P = preds[s]["task_y"], preds[s]["task_prob"]
        groups = df.loc[preds[s]["keys"], "accession"].values
        m = all_metrics(Y, P, thr, tb, cur["eval"]["spec_target"])
        ci = bootstrap_ci(Y, P, groups, thr, tb, cur["eval"]["spec_target"], n_boot // 5, seed)
        report["tasks"][s] = {k: dict(auroc=m[k]["auroc"], auroc_ci=ci[k]["auroc"], auprc=m[k]["auprc"],
                                      auprc_ci=ci[k]["auprc"], recall=m[k]["recall"],
                                      specificity=m[k]["specificity"], prevalence=float(Y[:, i].mean()))
                              for i, k in enumerate(tb)}

    # regression heads
    for j, name in enumerate(tr):
        scale = float(cfg["tasks"]["regression"][name][1])
        report["regression"][name] = {}
        for s in ("val", "test"):
            y = preds[s]["reg_y"][:, j] * scale
            yh = np.clip(preds[s]["reg_pred"][:, j] * scale, 0, None)
            bins = lambda v: np.digitize(v, [1e-6, 5.0 + 1e-6])          # noqa: E731  0 | (0,5] | >5
            report["regression"][name][s] = dict(
                mae=float(np.abs(y - yh).mean()), mae_pos=float(np.abs(y - yh)[y > 0].mean()),
                mae_gt5=float(np.abs(y - yh)[y > 5].mean()), pearson_r=float(np.corrcoef(y, yh)[0, 1]),
                bin3_accuracy=float((bins(y) == bins(yh)).mean()))

    # Marshall from the heads, and from the radiologist's own component labels
    idx = {k: i for i, k in enumerate(tb)}
    vk = preds["val"]["keys"]
    vt = df.loc[vk, "Marshall CT score"].values.astype(int)
    vkeep = vt != 5
    rule_thr, val_kappa = tune_rule_thresholds(preds["val"]["task_prob"][vkeep], idx, vt[vkeep],
                                               {h: float(thr[idx[h]]) for h, _ in RULE})
    report["rule_thresholds_tuned_on_val"] = dict(thresholds=rule_thr, val_kappa=val_kappa)
    for s in ("val", "test"):
        keys = preds[s]["keys"]
        truth = df.loc[keys, "Marshall CT score"].values.astype(int)
        keep = truth != 5
        groups = df.loc[keys, "accession"].values[keep]
        P, Y = preds[s]["task_prob"], preds[s]["task_y"]
        pred_f1 = marshall_rule({h: P[:, idx[h]] >= thr[idx[h]] for h, _ in RULE})
        pred = marshall_rule({h: P[:, idx[h]] >= rule_thr[h] for h, _ in RULE})
        comp = marshall_rule({h: Y[:, idx[h]] == 1 for h, _ in RULE})
        report["marshall"][s] = dict(
            excluded_category_v=int((~keep).sum()),
            model=marshall_metrics(truth[keep], pred[keep], groups, n_boot // 5, seed),
            model_per_head_f1_thresholds=marshall_metrics(truth[keep], pred_f1[keep], groups, n_boot // 5, seed),
            radiologist_components=marshall_metrics(truth[keep], comp[keep], groups, n_boot // 5, seed))
        plot_confusion(report["marshall"][s]["model"]["confusion"],
                       f"{args.run} {s}: Marshall from model heads (rule thresholds tuned on val)",
                       p(cfg, "plots") / "eval" / args.run / f"{s}_marshall_confusion.png")

    (run_dir / "marshall_report.json").write_text(json.dumps(report, indent=1))
    for s in ("val", "test"):
        print(f"\n=== {s} ===")
        for k, v in report["tasks"][s].items():
            print(f"  {k:16s} AUROC {v['auroc']:.3f} [{v['auroc_ci'][0]:.2f},{v['auroc_ci'][1]:.2f}] "
                  f"AUPRC {v['auprc']:.3f}  prev {v['prevalence']:.3f}")
        for name in tr:
            r = report["regression"][name][s]
            print(f"  {name}: MAE {r['mae']:.2f} mm (shifted scans {r['mae_pos']:.2f}, >5 mm {r['mae_gt5']:.2f}), "
                  f"r {r['pearson_r']:.3f}, 3-bin acc {r['bin3_accuracy']:.3f}")
        mm, rc = report["marshall"][s]["model"], report["marshall"][s]["radiologist_components"]
        mf = report["marshall"][s]["model_per_head_f1_thresholds"]
        print(f"  Marshall (model heads, per-head F1 thresholds): acc {mf['accuracy']:.3f}, "
              f"quadratic kappa {mf['kappa_quadratic']:.3f}")
        print(f"  Marshall (model heads, rule thresholds tuned on val): acc {mm['accuracy']:.3f}, quadratic kappa "
              f"{mm['kappa_quadratic']:.3f} [{mm['kappa_quadratic_ci95'][0]:.3f},{mm['kappa_quadratic_ci95'][1]:.3f}], "
              f"recall { {k: round(v, 3) for k, v in mm['recall'].items()} }")
        print("  confusion (rows radiologist I,II,III,IV,VI):", mm["confusion"])
        print(f"  Marshall (radiologist components, ceiling): acc {rc['accuracy']:.3f}, "
              f"quadratic kappa {rc['kappa_quadratic']:.3f}")


if __name__ == "__main__":
    main()
