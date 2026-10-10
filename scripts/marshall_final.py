"""Marshall CT score from three sources: the multi-task model (e2e_vh_mt), the basal-cistern model (cistern_vh)
and the HU lesion volumes (lesion_volume.py).

Components:
  VI   large lesion   : logistic regression on [mt marshall_vi logit, log1p largest / total / extra / intra cm3],
                         fit on val (5-fold patient-grouped out-of-fold scores on val for tuning, full-val fit for test)
  IV   shift > 5 mm   : mt head mls_gt5
  III  cisterns       : mean logit of the mt head and the cistern model head (cist_compressed)
  II   any pathology  : mt head marshall_ge2
Rule (guideline priority VI > IV > III > II > I) with the 4 thresholds tuned jointly on val (max quadratic kappa),
plus a learned multinomial combiner over all features. Category V excluded. Test is touched once.

  python scripts/marshall_final.py --mt e2e_vh_mt --cistern cistern_vh
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_table, task_names  # noqa: E402
from scripts.eval_marshall import (CATS, RULE, marshall_metrics, marshall_rule, plot_confusion,  # noqa: E402
                                   tune_rule_thresholds)
from utils.config import load_config, p  # noqa: E402

VOL_COLS = ["largest_cm3", "total_cm3", "extra_axial_cm3", "intra_axial_cm3"]


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def load_preds(run_dir, split):
    z = np.load(run_dir / f"preds_{split}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}, [str(k) for k in z["keys"]]


def oof_and_full(X, y, groups, C=1.0):
    """Out-of-fold probabilities on X (for tuning on val) and a model fit on all of X (for test)."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    sc = StandardScaler().fit(X)
    return oof, (sc, LogisticRegression(C=C, max_iter=4000).fit(sc.transform(X), y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt", default="e2e_vh_mt")
    ap.add_argument("--cistern", default="cistern_vh")
    ap.add_argument("--cistern-outputs", default="/mnt/d/All parts NIFTI files/ich_pipeline/outputs_cistern")
    args = ap.parse_args()
    cur = load_config()
    out_root = p(cur, "outputs")
    mt_dir, cs_dir = out_root / "runs" / args.mt, Path(args.cistern_outputs) / "runs" / args.cistern
    mcfg = json.loads((mt_dir / "config.json").read_text())
    tb, _ = task_names(mcfg)
    ti = {k: i for i, k in enumerate(tb)}
    df = load_table(cur).set_index("key")

    S = {}
    for s in ("val", "test"):
        mt, keys = load_preds(mt_dir, s)
        cs, ckeys = load_preds(cs_dir, s)
        cpos = {k: i for i, k in enumerate(ckeys)}
        cs_prob = cs["task_prob"][[cpos[k] for k in keys]]
        vol = pd.read_csv(out_root / "lesion_volume" / f"volumes_{s}.csv").set_index("key")
        V = np.log1p(vol.loc[keys, VOL_COLS].fillna(0).clip(lower=0).values)
        truth = df.loc[keys, "Marshall CT score"].values.astype(int)
        keep = truth != 5
        S[s] = dict(keys=np.array(keys)[keep], truth=truth[keep], groups=df.loc[keys, "accession"].values[keep],
                    mt=mt["task_prob"][keep], mt_cls=mt["prob"][keep], mt_reg=mt["reg_pred"][keep],
                    cs=cs_prob[keep], V=V[keep], y_task=mt["task_y"][keep])
    rep = dict(mt=args.mt, cistern=args.cistern, components={})

    # VI: head + volumes
    def vi_X(d):
        return np.column_stack([logit(d["mt"][:, ti["marshall_vi"]]), d["V"]])
    yv_vi = S["val"]["y_task"][:, ti["marshall_vi"]]
    vi_oof, (vsc, vmod) = oof_and_full(vi_X(S["val"]), yv_vi, S["val"]["groups"])
    vi = {"val": vi_oof, "test": vmod.predict_proba(vsc.transform(vi_X(S["test"])))[:, 1]}
    # cisterns: average of the two heads
    cist = {s: 1 / (1 + np.exp(-0.5 * (logit(S[s]["mt"][:, ti["cist_compressed"]]) +
                                       logit(S[s]["cs"][:, ti["cist_compressed"]])))) for s in S}
    for s in ("val", "test"):
        y = S[s]["y_task"]
        rep["components"][s] = dict(
            vi_head=roc_auc_score(y[:, ti["marshall_vi"]], S[s]["mt"][:, ti["marshall_vi"]]),
            vi_volume_largest=roc_auc_score(y[:, ti["marshall_vi"]], S[s]["V"][:, 0]),
            vi_head_plus_volume=roc_auc_score(y[:, ti["marshall_vi"]], vi[s]),
            cist_mt_head=roc_auc_score(y[:, ti["cist_compressed"]], S[s]["mt"][:, ti["cist_compressed"]]),
            cist_cistern_model=roc_auc_score(y[:, ti["cist_compressed"]], S[s]["cs"][:, ti["cist_compressed"]]),
            cist_both=roc_auc_score(y[:, ti["cist_compressed"]], cist[s]),
            cist_effaced_cistern_model=roc_auc_score(y[:, ti["cist_effaced"]], S[s]["cs"][:, ti["cist_effaced"]]))

    # rule with tuned thresholds
    def comps(s):
        d = S[s]
        cols = {"marshall_vi": vi[s], "mls_gt5": d["mt"][:, ti["mls_gt5"]], "cist_compressed": cist[s],
                "marshall_ge2": d["mt"][:, ti["marshall_ge2"]]}
        return np.column_stack([cols[h] for h, _ in RULE]), {h: j for j, (h, _) in enumerate(RULE)}
    Pv, idx = comps("val")
    thr, vk = tune_rule_thresholds(Pv, idx, S["val"]["truth"], {h: 0.5 for h, _ in RULE})
    rep["rule_thresholds"] = dict(thresholds=thr, val_kappa=vk)
    Pt, _ = comps("test")
    pred_rule = marshall_rule({h: Pt[:, idx[h]] >= thr[h] for h, _ in RULE})

    # learned combiner over everything
    def all_X(s):
        d = S[s]
        return np.column_stack([logit(d["mt"]), logit(d["mt_cls"]), d["mt_reg"] * 10, logit(d["cs"][:, [ti["cist_compressed"], ti["cist_effaced"]]]),
                                d["V"], logit(vi[s])])
    sc = StandardScaler().fit(all_X("val"))
    best = None
    for C in (0.03, 0.1, 0.3, 1.0):
        oof = np.zeros_like(S["val"]["truth"])
        for tr, te in GroupKFold(5).split(all_X("val"), S["val"]["truth"], S["val"]["groups"]):
            s2 = StandardScaler().fit(all_X("val")[tr])
            m = LogisticRegression(C=C, max_iter=4000).fit(s2.transform(all_X("val")[tr]), S["val"]["truth"][tr])
            oof[te] = m.predict(s2.transform(all_X("val")[te]))
        k = cohen_kappa_score(S["val"]["truth"], oof, weights="quadratic")
        if best is None or k > best[0]:
            best = (k, C)
    comb = LogisticRegression(C=best[1], max_iter=4000).fit(sc.transform(all_X("val")), S["val"]["truth"])
    pred_comb = comb.predict(sc.transform(all_X("test")))
    rep["combiner"] = dict(C=best[1], val_cv_kappa=float(best[0]))

    t, g = S["test"]["truth"], S["test"]["groups"]
    rep["test"] = dict(rule=marshall_metrics(t, pred_rule, g, 200, 42), combiner=marshall_metrics(t, pred_comb, g, 200, 42))
    (out_root / "runs" / args.mt / "marshall_final_report.json").write_text(json.dumps(rep, indent=1, default=float))
    plot_confusion(rep["test"]["rule"]["confusion"], "test: Marshall rule (mt + cistern model + volume, tuned on val)",
                   p(cur, "plots") / "eval" / args.mt / "test_marshall_final_rule_confusion.png")
    plot_confusion(rep["test"]["combiner"]["confusion"], "test: Marshall learned combiner (mt + cistern + volume)",
                   p(cur, "plots") / "eval" / args.mt / "test_marshall_final_combiner_confusion.png")
    print("component AUROCs:", json.dumps(rep["components"], indent=1, default=lambda v: round(float(v), 3)))
    for name in ("rule", "combiner"):
        m = rep["test"][name]
        print(f"TEST {name}: acc {m['accuracy']:.3f}, quadratic kappa {m['kappa_quadratic']:.3f} "
              f"[{m['kappa_quadratic_ci95'][0]:.3f},{m['kappa_quadratic_ci95'][1]:.3f}], "
              f"recall { {a: round(b, 3) for a, b in m['recall'].items()} }")
        print("  confusion (rows radiologist I,II,III,IV,VI):", m["confusion"])


if __name__ == "__main__":
    main()
