"""Data-leakage audit (CPU only).

1. Split integrity: patient (accession) overlap, duplicate scan keys.
2. Exact duplicate files: identical NIfTI byte size -> md5 among those groups.
3. Near-duplicate scans across splits: cosine similarity of mean-pooled frozen VisionHOPE features,
   compared with same-patient (different CT) and unrelated-pair similarity; flagged pairs checked for Age/Sex.
4. Metadata shortcuts: can acquisition/file metadata alone predict the labels (val AUROC)?
5. Shuffled-label control: features + permuted train labels must give val AUROC ~0.5.
Writes outputs/leakage_audit/report.json and a similarity plot.
"""
import hashlib
import json
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from data.dataset import load_table  # noqa: E402
from utils.config import load_config, p  # noqa: E402

cfg = load_config(None)
classes = cfg["classes"]
out = p(cfg, "outputs") / "leakage_audit"
out.mkdir(parents=True, exist_ok=True)
rep = {}

df = load_table(cfg).reset_index(drop=True)
print(f"{len(df)} scans, splits: {df['split'].value_counts().to_dict()}")

# 1 ---------------------------------------------------------------- split integrity
pat = {s: set(df.loc[df["split"] == s, "accession"]) for s in ("train", "val", "test")}
rep["patient_overlap"] = {f"{a}-{b}": len(pat[a] & pat[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
rep["duplicate_keys"] = int(df["key"].duplicated().sum())
rep["patients_in_multiple_splits"] = int(df.groupby("accession")["split"].nunique().gt(1).sum())
print("1. patient overlap:", rep["patient_overlap"], "| dup keys", rep["duplicate_keys"],
      "| patients in >1 split", rep["patients_in_multiple_splits"])

# 2 ---------------------------------------------------------------- exact duplicate files
groups = df.groupby("file_bytes")["key"].apply(list)
groups = groups[groups.map(len) > 1]
md5 = {}
for keys in groups:
    for k in keys:
        h = hashlib.md5()
        with open(df.loc[df["key"] == k, "nifti_path"].iloc[0], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        md5[k] = h.hexdigest()
by_hash = defaultdict(list)
for k, h in md5.items():
    by_hash[h].append(k)
dups = [v for v in by_hash.values() if len(v) > 1]
split_of = dict(zip(df["key"], df["split"]))
acc_of = dict(zip(df["key"], df["accession"]))
rep["exact_duplicate_groups"] = len(dups)
pd.DataFrame([dict(key=k, duplicate_of=sorted(v)[0]) for v in dups for k in sorted(v)[1:]]).to_csv(
    out / "exact_duplicates.csv", index=False)
rep["exact_duplicates_cross_split"] = [v for v in dups if len({split_of[k] for k in v}) > 1]
rep["exact_duplicates_cross_patient"] = [v for v in dups if len({acc_of[k] for k in v}) > 1]
print(f"2. same-size groups {len(groups)}, exact duplicate groups {len(dups)}, cross-split "
      f"{len(rep['exact_duplicates_cross_split'])}, cross-patient {len(rep['exact_duplicates_cross_patient'])}")

# 3 ---------------------------------------------------------------- near duplicates
fdir = p(cfg, "feature_cache") / "visionhope_tiny" / "clean"
F = np.stack([np.load(fdir / f"{k}.npy").astype(np.float32).mean(0) for k in df["key"]])
Fm = F - F[df["split"].values == "train"].mean(0)
Fn = Fm / np.linalg.norm(Fm, axis=1, keepdims=True)
tr = np.flatnonzero(df["split"].values == "train")
ev = np.flatnonzero(df["split"].values != "train")
S = Fn[ev] @ Fn[tr].T
nn_idx = S.argmax(1)
nn_sim = S.max(1)

acc = df["accession"].values
same_pat = []
for a, idx in df.groupby("accession").indices.items():
    if len(idx) > 1:
        sub = Fn[idx] @ Fn[idx].T
        same_pat += sub[np.triu_indices(len(idx), 1)].tolist()
same_pat = np.array(same_pat)
rng = np.random.default_rng(0)
i, j = rng.integers(0, len(df), 20000), rng.integers(0, len(df), 20000)
keep = acc[i] != acc[j]
unrel = np.einsum("nd,nd->n", Fn[i[keep]], Fn[j[keep]])
# Null for "best match among ~6.6k other patients": train scans vs train scans of other patients.
q = rng.choice(tr, min(len(ev), len(tr)), replace=False)
Sq = Fn[q] @ Fn[tr].T
Sq[acc[q][:, None] == acc[tr][None, :]] = -2
null_nn = Sq.max(1)
thr = float(np.percentile(null_nn, 99.5))
flag = np.flatnonzero(nn_sim >= thr)
age, sex = df["Age"].values, df["Sex"].values
flagged = []
for f in flag:
    a, b = ev[f], tr[nn_idx[f]]
    flagged.append(dict(eval_key=df["key"][a], eval_split=df["split"][a], train_key=df["key"][b], sim=float(nn_sim[f]),
                        same_age=bool(age[a] == age[b]), same_sex=bool(sex[a] == sex[b]),
                        same_labels=bool((df.loc[a, classes].values == df.loc[b, classes].values).all())))
fl = pd.DataFrame(flagged)
fl.to_csv(out / "near_duplicate_candidates.csv", index=False)
rep["near_dup"] = dict(
    same_patient_sim_median=float(np.median(same_pat)), same_patient_sim_p5=float(np.percentile(same_pat, 5)),
    unrelated_sim_median=float(np.median(unrel)), unrelated_sim_p99=float(np.percentile(unrel, 99)),
    unrelated_sim_p999=float(np.percentile(unrel, 99.9)),
    eval_nn_to_train_sim_median=float(np.median(nn_sim)), eval_nn_to_train_sim_max=float(nn_sim.max()),
    null_train_nn_other_patient_median=float(np.median(null_nn)), null_train_nn_other_patient_max=float(null_nn.max()),
    frac_same_patient_pairs_above_eval_max=float((same_pat > nn_sim.max()).mean()),
    threshold_null_p995=thr, n_eval_scans=int(len(ev)), n_flagged=int(len(fl)),
    expected_flagged_by_chance=round(0.005 * len(ev), 1),
    n_flagged_same_age_and_sex=int((fl["same_age"] & fl["same_sex"]).sum()) if len(fl) else 0,
    n_eval_sim_ge_0_99=int((nn_sim >= 0.99).sum()))
print("3.", rep["near_dup"])
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(-0.2, 1.0, 121)
ax.hist(unrel, bins, density=True, alpha=0.5, label="unrelated pairs")
ax.hist(same_pat, bins, density=True, alpha=0.5, label="same patient, different CT")
ax.hist(nn_sim, bins, density=True, alpha=0.5, label="val/test scan -> nearest train scan")
ax.hist(null_nn, bins, density=True, alpha=0.5, histtype="step", lw=1.5, color="k",
        label="train scan -> nearest other-patient train scan (null)")
ax.axvline(thr, color="k", ls="--", lw=1, label="null 99.5th pct")
ax.set_xlabel("cosine similarity (mean-pooled VisionHOPE features)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "near_duplicate_similarity.png", dpi=110)

# 4 ---------------------------------------------------------------- metadata shortcuts
df["folder_num"] = pd.to_numeric(df["folder"], errors="coerce")
meta_cols = ["n_slices", "slice_thickness", "file_bytes", "folder_num", "ct_index", "n_kept", "crop_mm", "n_ring_slices"]
df["px"] = df["pixel_spacing"].astype(str).str.extract(r"([\d.]+)").astype(float)
meta_cols.append("px")
X = df[meta_cols].astype(float).fillna(-1).values
is_tr, is_va = df["split"].values == "train", df["split"].values == "val"
meta = {}
for c in classes:
    y = df[c].values
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[is_tr], y[is_tr])
    meta[c] = round(float(roc_auc_score(y[is_va], m.predict_proba(X[is_va])[:, 1])), 3)
single = {col: {c: round(float(abs(roc_auc_score(df.loc[is_va, c], df.loc[is_va, col].astype(float).fillna(-1)) - 0.5) + 0.5), 3)
                for c in classes} for col in meta_cols}
rep["metadata_val_auroc_all_cols"] = meta
rep["metadata_val_auroc_single_col_max_direction"] = single
print("4. metadata-only val AUROC:", meta)
for col, v in single.items():
    print(f"   {col:16s}", v)

# 5 ---------------------------------------------------------------- shuffled labels
Y = df[classes].values
perm = rng.permutation(np.flatnonzero(is_tr))
Ysh = Y.copy()
Ysh[np.flatnonzero(is_tr)] = Y[perm]
real, shuf = {}, {}
for k, c in enumerate(classes):
    for target, store in ((Y, real), (Ysh, shuf)):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
        m.fit(F[is_tr], target[is_tr, k])
        store[c] = round(float(roc_auc_score(Y[is_va, k], m.predict_proba(F[is_va])[:, 1])), 3)
rep["features_real_labels_val_auroc"] = real
rep["features_shuffled_labels_val_auroc"] = shuf
print("5. real labels  :", real, "macro", round(np.mean(list(real.values())), 3))
print("   shuffled     :", shuf, "macro", round(np.mean(list(shuf.values())), 3))

(out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
print("report ->", out / "report.json")
