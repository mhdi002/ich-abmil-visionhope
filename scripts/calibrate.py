"""Fit the decision layer of a trained run on the val split and save it as <run>/calibration.json, so that
predict_scan.py can score a new scan without the dataset.

Saved (all fit on val only):
  class_thresholds   5 hemorrhage classes, max F1
  task_thresholds    every binary head, max F1 (midline shift > 0 / > 5 mm, cisterns, ...)
  vi_model           logistic regression for Marshall VI on [VI head logit, log1p lesion volumes] (lesion_volume.py)
  rule_thresholds    the 4 Marshall rule thresholds (VI, shift > 5 mm, cisterns compressed, any pathology),
                     tuned jointly for quadratic kappa vs the radiologist score
Reports the val numbers and, once, the test numbers of the calibrated rule.

  python scripts/calibrate.py --run vh_small_all
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_table, task_names  # noqa: E402
from eval import choose_thresholds  # noqa: E402
from scripts.eval_marshall import RULE, marshall_metrics, marshall_rule, tune_rule_thresholds  # noqa: E402
from utils.config import load_config, p, parse_overrides  # noqa: E402

VOL_COLS = ["largest_cm3", "total_cm3", "extra_axial_cm3", "intra_axial_cm3"]


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def vi_features(vi_prob, vols):
    return np.column_stack([logit(vi_prob), np.log1p(np.clip(vols, 0, None))])


def vi_prob_from(cal, vi_prob, vols):
    """Apply the saved VI model (used by predict_scan.py too)."""
    m = cal["vi_model"]
    x = (vi_features(np.atleast_1d(vi_prob), np.atleast_2d(vols)) - np.array(m["mean"])) / np.array(m["scale"])
    return 1 / (1 + np.exp(-(x @ np.array(m["coef"]) + m["intercept"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cur = load_config(None, parse_overrides(args.set))
    run_dir = p(cur, "outputs") / "runs" / args.run
    cfg = json.loads((run_dir / "config.json").read_text())
    tb, tr = task_names(cfg)
    ti = {k: i for i, k in enumerate(tb)}
    df = load_table(cur).set_index("key")
    vol_dir = p(cur, "outputs") / "lesion_volume"

    S = {}
    for s in ("val", "test"):
        z = np.load(run_dir / f"preds_{s}.npz", allow_pickle=True)
        keys = [str(k) for k in z["keys"]]
        vol = pd.read_csv(vol_dir / f"volumes_{s}.csv").set_index("key").loc[keys, VOL_COLS].fillna(0).values
        S[s] = dict(keys=keys, y=z["y"], prob=z["prob"], task_y=z["task_y"], task_prob=z["task_prob"], vol=vol,
                    truth=df.loc[keys, "Marshall CT score"].values.astype(int), groups=df.loc[keys, "accession"].values)
    v = S["val"]
    cal = dict(run=args.run, classes=cfg["classes"], tasks=tb, regression=tr,
               regression_scale={k: float(cfg["tasks"]["regression"][k][1]) for k in tr},
               class_thresholds=dict(zip(cfg["classes"], map(float, choose_thresholds(v["y"], v["prob"], "f1")))),
               task_thresholds=dict(zip(tb, map(float, choose_thresholds(v["task_y"], v["task_prob"], "f1")))))

    X = vi_features(v["task_prob"][:, ti["marshall_vi"]], v["vol"])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=4000).fit(sc.transform(X), v["task_y"][:, ti["marshall_vi"]])
    cal["vi_model"] = dict(features=["vi_head_logit"] + [f"log1p_{c}" for c in VOL_COLS], mean=sc.mean_.tolist(),
                           scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]))

    def comps(d):
        vi = vi_prob_from(cal, d["task_prob"][:, ti["marshall_vi"]], d["vol"])
        cols = {"marshall_vi": vi, "mls_gt5": d["task_prob"][:, ti["mls_gt5"]],
                "cist_compressed": d["task_prob"][:, ti["cist_compressed"]],
                "marshall_ge2": d["task_prob"][:, ti["marshall_ge2"]]}
        return np.column_stack([cols[h] for h, _ in RULE]), {h: j for j, (h, _) in enumerate(RULE)}
    keep = v["truth"] != 5
    Pv, idx = comps({k: (x[keep] if isinstance(x, np.ndarray) else x) for k, x in v.items()})
    thr, vk = tune_rule_thresholds(Pv, idx, v["truth"][keep], {h: 0.5 for h, _ in RULE})
    cal["rule_thresholds"] = thr
    cal["rule_order"] = [[h, c] for h, c in RULE]

    t = S["test"]
    tk = t["truth"] != 5
    Pt, _ = comps({k: (x[tk] if isinstance(x, np.ndarray) else x) for k, x in t.items()})
    pred = marshall_rule({h: Pt[:, idx[h]] >= thr[h] for h, _ in RULE})
    m = marshall_metrics(t["truth"][tk], pred, t["groups"][tk], 200, 42)
    vi_t = vi_prob_from(cal, t["task_prob"][:, ti["marshall_vi"]], t["vol"])
    cal["validation"] = dict(val_rule_kappa=vk, test_rule=dict(accuracy=m["accuracy"], kappa_quadratic=m["kappa_quadratic"],
                                                              kappa_ci95=m["kappa_quadratic_ci95"], recall=m["recall"]),
                             test_vi_auroc_head=float(roc_auc_score(t["task_y"][:, ti["marshall_vi"]],
                                                                    t["task_prob"][:, ti["marshall_vi"]])),
                             test_vi_auroc_head_plus_volume=float(roc_auc_score(t["task_y"][:, ti["marshall_vi"]], vi_t)))
    (run_dir / "calibration.json").write_text(json.dumps(cal, indent=1))
    print(json.dumps({k: cal[k] for k in ("class_thresholds", "rule_thresholds", "validation")}, indent=1))
    print("saved", run_dir / "calibration.json")


if __name__ == "__main__":
    main()
