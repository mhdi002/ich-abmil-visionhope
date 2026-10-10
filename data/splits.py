"""Phase 2: patient-level multi-label iterative stratification (70/15/15) + 5-fold CV."""
import argparse
import json

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

from utils.config import load_config, p, seed_everything


def splits_from_preds(pred_dir):
    """key -> split from the keys saved in <pred_dir>/preds_{train,val,test}.npz of an earlier run."""
    from pathlib import Path
    out = {}
    for s in ("train", "val", "test"):
        for k in np.load(Path(pred_dir) / f"preds_{s}.npz", allow_pickle=True)["keys"]:
            out[str(k)] = s
    return pd.Series(out)


def build_splits(cfg, from_preds=None):
    """from_preds: reuse the exact split assignment of an earlier run (scans it did not use are dropped)."""
    seed = cfg["seed"]
    seed_everything(seed)
    classes = cfg["classes"]
    from data.dataset import load_table
    m = load_table(cfg, require_split=False)

    pat = m.groupby("accession")[classes].max()
    pat["later_operated"] = m.groupby("accession")["later_operated"].max().astype(int)
    strat_cols = classes + ["later_operated"]
    acc = pat.index.values
    Y = pat[strat_cols].values

    f_tr, f_va, f_te = cfg["splits"]["fractions"]
    s1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=f_va + f_te, random_state=seed)
    tr, rest = next(s1.split(acc, Y))
    s2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=f_te / (f_va + f_te), random_state=seed)
    va, te = next(s2.split(acc[rest], Y[rest]))
    split = pd.Series("train", index=acc)
    split.iloc[rest[va]] = "val"
    split.iloc[rest[te]] = "test"

    kf = MultilabelStratifiedKFold(n_splits=cfg["splits"]["n_folds"], shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=acc)
    for k, (_, idx) in enumerate(kf.split(acc, Y)):
        fold.iloc[idx] = k

    m["split"] = m["accession"].map(split)
    if from_preds:
        prev = splits_from_preds(from_preds)
        missing = sorted(set(prev.index) - set(m["key"]))
        print(f"reusing splits of {from_preds}: {len(prev)} scans, {len(missing)} no longer available "
              f"{missing[:10]}, {int((~m['key'].isin(prev.index)).sum())} new scans dropped")
        m = m[m["key"].isin(prev.index)].copy()
        m["split"] = m["key"].map(prev)
    m["fold"] = m["accession"].map(fold)
    dt = pd.to_datetime(m["study_datetime"])
    m["chrono_holdout"] = dt >= dt.quantile(0.9)

    sets = {s: set(m.loc[m["split"] == s, "accession"]) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert m.groupby("accession")["split"].nunique().max() == 1
    assert m.groupby("accession")["fold"].nunique().max() == 1

    out = m[["key", "accession", "ct_index", "split", "fold", "chrono_holdout"]]
    out.to_csv(p(cfg, "splits"), index=False)

    report = {}
    for s in ("train", "val", "test"):
        d = m[m["split"] == s]
        report[s] = dict(patients=int(d["accession"].nunique()), scans=int(len(d)),
                         positives={c: int(d[c].sum()) for c in classes},
                         prevalence={c: round(float(d[c].mean()), 4) for c in classes},
                         patients_with_positive={c: int(d.groupby("accession")[c].max().sum()) for c in classes},
                         later_operated_scans=int(d["later_operated"].sum()))
    report["folds"] = {int(k): dict(patients=int(g["accession"].nunique()), scans=int(len(g)),
                                    positives={c: int(g[c].sum()) for c in classes})
                       for k, g in m.groupby("fold")}
    report["patient_overlap"] = 0
    (p(cfg, "outputs") / "phase0" / "splits_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({s: report[s] for s in ("train", "val", "test")}, indent=1))

    from utils import viz
    viz.plot_class_distribution({s: m.loc[m["split"] == s, classes].values for s in ("train", "val", "test")},
                                classes, p(cfg, "plots") / "dataset" / "class_distribution_splits.png")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--from-preds", default=None, help="run dir with preds_{train,val,test}.npz to copy splits from")
    args = ap.parse_args()
    build_splits(load_config(args.config), args.from_preds)


if __name__ == "__main__":
    main()
