"""Learned Marshall combiner: multinomial logistic regression over the multi-task model outputs.

Inputs per scan: logits of the 17 task heads + 5 hemorrhage classes + the midline-shift regression (mm).
Fit on the val split only (5-fold patient-grouped CV on val to pick C), refit on all val, evaluate once on test.
Category V (10 cohort scans, surgically evacuated) is excluded as in eval_marshall.py.

  python scripts/marshall_stack.py --run e2e_vh_mt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_table  # noqa: E402
from scripts.eval_marshall import CATS, kappa_ci, plot_confusion  # noqa: E402
from utils.config import load_config, p  # noqa: E402


def features(z):
    logit = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1))  # noqa: E731
    return np.concatenate([logit(z["task_prob"]), logit(z["prob"]), z["reg_pred"] * 10.0], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e2e_vh_mt")
    args = ap.parse_args()
    cur = load_config()
    run_dir = p(cur, "outputs") / "runs" / args.run
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg["paths"] = {**cfg["paths"], **cur["paths"]}
    df = load_table(cfg).set_index("key")
    data = {}
    for s in ("val", "test"):
        z = np.load(run_dir / f"preds_{s}.npz", allow_pickle=True)
        keys = [str(k) for k in z["keys"]]
        y = df.loc[keys, "Marshall CT score"].values.astype(int)
        keep = y != 5
        data[s] = dict(X=features(z)[keep], y=y[keep], g=df.loc[keys, "accession"].values[keep])

    Xv, yv, gv = data["val"]["X"], data["val"]["y"], data["val"]["g"]
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        for cw in (None, "balanced"):
            oof = np.zeros_like(yv)
            for tr, te in GroupKFold(5).split(Xv, yv, gv):
                sc = StandardScaler().fit(Xv[tr])
                m = LogisticRegression(C=C, class_weight=cw, max_iter=4000).fit(sc.transform(Xv[tr]), yv[tr])
                oof[te] = m.predict(sc.transform(Xv[te]))
            k = cohen_kappa_score(yv, oof, weights="quadratic")
            acc = float((oof == yv).mean())
            print(f"  C={C:<5} class_weight={cw!s:8s} val CV: quadratic kappa {k:.3f}, acc {acc:.3f}")
            if best is None or k > best[0]:
                best = (k, C, cw, acc)
    k, C, cw, acc = best
    sc = StandardScaler().fit(Xv)
    m = LogisticRegression(C=C, class_weight=cw, max_iter=4000).fit(sc.transform(Xv), yv)
    Xt, yt, gt = data["test"]["X"], data["test"]["y"], data["test"]["g"]
    pt = m.predict(sc.transform(Xt))
    cm = confusion_matrix(yt, pt, labels=CATS)
    rep = dict(chosen=dict(C=C, class_weight=cw, val_cv_kappa=k, val_cv_acc=acc),
               test=dict(n=int(len(yt)), accuracy=float((pt == yt).mean()),
                         kappa_quadratic=float(cohen_kappa_score(yt, pt, weights="quadratic")),
                         kappa_quadratic_ci95=kappa_ci(yt, pt, gt, 200, cfg["seed"]),
                         recall={f"cat{c}": float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(CATS)},
                         confusion=cm.tolist()))
    (run_dir / "marshall_stack_report.json").write_text(json.dumps(rep, indent=1))
    plot_confusion(cm, f"{args.run} test: Marshall from learned combiner (fit on val)",
                   p(cfg, "plots") / "eval" / args.run / "test_marshall_stack_confusion.png")
    t = rep["test"]
    print(f"chosen C={C} class_weight={cw} (val CV kappa {k:.3f})")
    print(f"TEST: acc {t['accuracy']:.3f}, quadratic kappa {t['kappa_quadratic']:.3f} "
          f"[{t['kappa_quadratic_ci95'][0]:.3f},{t['kappa_quadratic_ci95'][1]:.3f}], "
          f"recall { {a: round(b, 3) for a, b in t['recall'].items()} }")
    print("confusion (rows radiologist I,II,III,IV,VI):", t["confusion"])


if __name__ == "__main__":
    main()
