"""Phase 0.1-0.2: label file -> pre-surgery cohort -> NIfTI join -> scan_manifest.csv."""
import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from utils.config import ACC, IDX, NEW_SURGERY, NON_TBI, ROOT, load_config, p

NAME_RE = re.compile(r"^(\d+)_(\d+)\.nii(\.gz)?$")
EXCLUDE_TEXT_RE = re.compile(r"catheter|cranioplast|burr", re.I)
SPEC_PRE = {"EDH": 1496, "SDH": 3038, "tSAH": 2552, "Has_IVH": 1008, "Contusion": 5043}
SPEC_FULL = {"EDH": 1878, "SDH": 4255, "tSAH": 3517, "Has_IVH": 1695, "Contusion": 7025}


def read_table(xlsx, pickle_path):
    pickle_path = Path(pickle_path)
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    df = pd.read_excel(xlsx)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(pickle_path)
    return df


def header_info(path):
    try:
        img = nib.load(path)
        codes = nib.aff2axcodes(img.affine)
        affine_axial = next(i for i, c in enumerate(codes) if c in ("S", "I"))
        # The short axis is the acquired axial stack; some affines mislabel it (e.g. LSP on 512x512x30).
        axial = int(np.argmin(img.shape[:3]))
        zooms = img.header.get_zooms()[:3]
        inplane = [zooms[i] for i in range(3) if i != axial]
        return dict(nifti_path=path, shape=str(tuple(img.shape)), axcodes="".join(codes),
                    slice_axis=axial, affine_consistent=bool(affine_axial == axial),
                    n_slices=int(img.shape[axial]), slice_thickness=float(zooms[axial]),
                    pixel_spacing=float(np.mean(inplane)), header_ok=True)
    except Exception as e:  # noqa: BLE001 - logged in the manifest
        return dict(nifti_path=path, header_ok=False, header_error=str(e)[:200])


def scan_files(cfg):
    root = p(cfg, "nifti_root")
    rows, bad = [], []
    for sub in cfg["paths"]["nifti_subdirs"]:
        for f in sorted((root / sub).iterdir()):
            m = NAME_RE.match(f.name)
            if m:
                rows.append(dict(accession=int(m.group(1)), ct_index=int(m.group(2)),
                                 nifti_path=str(f), folder=sub, file_bytes=f.stat().st_size))
            else:
                bad.append(str(f))
    return pd.DataFrame(rows), bad


def apply_cohort(df, log):
    df = df.sort_values([ACC, IDX]).reset_index(drop=True)
    hs = df.groupby(ACC)["Has_Surgery"]
    drops = df.groupby(ACC)["Has_Surgery"].apply(lambda s: bool((s.diff() < 0).any()))
    log["patients_has_surgery_1_to_0"] = int(drops.sum())
    log["patients_ever_has_surgery"] = int((hs.max() == 1).sum())

    first_post = df[df["Has_Surgery"] == 1].groupby(ACC)[IDX].min()
    df = df.join(first_post.rename("first_post"), on=ACC)
    pre = df["first_post"].isna() | (df[IDX] < df["first_post"])
    log["cts_after_rule"] = int(pre.sum())
    log["post_op_cts_removed"] = int((~pre).sum())
    flagged = pre & ((df[NEW_SURGERY] == 1) | (df["Postsurgical subdural"] == 1))
    log["residual_flagged_rows_removed"] = int(flagged.sum())
    text = df[NON_TBI].fillna("").astype(str)
    text_hit = pre & ~flagged & text.str.contains(EXCLUDE_TEXT_RE)
    log["free_text_excluded_rows"] = int(text_hit.sum())
    log["free_text_excluded_values"] = sorted(text[text_hit].unique().tolist())
    df["is_pre_surgery"] = pre & ~flagged & ~text_hit
    ever = (df.groupby(ACC)["Has_Surgery"].max() == 1) | (df.groupby(ACC)[NEW_SURGERY].max() == 1)
    df["later_operated"] = df[ACC].map(ever)
    return df


def chronology_check(df, log):
    dt = pd.to_datetime(df["Study Dates"].astype(str) + " " + df["Study Times"].astype(str),
                        errors="coerce")
    df["study_datetime"] = dt
    log["unparseable_datetimes"] = int(dt.isna().sum())
    bad = []
    for acc, g in df.sort_values([ACC, IDX]).groupby(ACC):
        d = g["study_datetime"].dropna()
        if len(d) > 1 and (d.diff().dt.total_seconds() < 0).any():
            bad.append(int(acc))
    log["patients_non_chronological"] = len(bad)
    log["patients_non_chronological_frac"] = len(bad) / df[ACC].nunique()
    log["non_chronological_examples"] = bad[:20]
    contiguous = df.groupby(ACC)[IDX].apply(lambda s: sorted(s) == list(range(1, len(s) + 1)))
    log["patients_non_contiguous_index"] = int((~contiguous).sum())
    return df


def surgery_disagreement(df, pat, log):
    ever = df.groupby(ACC)["Has_Surgery"].max()
    m = pat.set_index(ACC)["Having first surgery"].reindex(ever.index)
    log["flagged_but_having_first_surgery_0"] = sorted(int(a) for a in ever[(ever == 1) & (m == 0)].index)
    log["having_first_surgery_1_but_never_flagged"] = sorted(int(a) for a in ever[(ever == 0) & (m == 1)].index)


def label_stats(df, classes):
    y = df[classes].values
    co = (y.T @ y).astype(int)
    return dict(
        n_cts=int(len(df)), n_patients=int(df[ACC].nunique()),
        positives={c: int(df[c].sum()) for c in classes},
        prevalence={c: round(float(df[c].mean()), 4) for c in classes},
        patients_with_positive={c: int(df.groupby(ACC)[c].max().sum()) for c in classes},
        labels_per_ct={int(k): int(v) for k, v in pd.Series(y.sum(1)).value_counts().sort_index().items()},
        cooccurrence=co.tolist(),
        ct_index_counts={str(k): int(v) for k, v in df[IDX].clip(upper=6).value_counts().sort_index().items()},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes = cfg["classes"]
    out_dir = p(cfg, "outputs") / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {}

    df = read_table(p(cfg, "ct_label_xlsx"), p(cfg, "outputs") / "cache" / "ct_level.pkl")
    pat = read_table(p(cfg, "patient_xlsx"), p(cfg, "outputs") / "cache" / "patient_level.pkl")
    log["label_rows"] = len(df)
    log["label_patients"] = int(df[ACC].nunique())
    log["duplicate_label_keys"] = int(df.duplicated([ACC, IDX]).sum())
    any5 = (df[classes].max(1) == 1).astype(int)
    log["has_any_ich_mismatch"] = int((any5 != df[cfg["aux_target"]]).sum())
    log["full_file_stats"] = label_stats(df, classes)
    log["full_file_vs_spec"] = {c: [log["full_file_stats"]["positives"][c], SPEC_FULL[c]] for c in classes}

    df = chronology_check(df, log)
    df = apply_cohort(df, log)
    surgery_disagreement(df, pat, log)

    files, bad_names = scan_files(cfg)
    log["nifti_files"] = len(files)
    log["nifti_bad_names"] = bad_names[:50]
    dup = files[files.duplicated(["accession", "ct_index"], keep=False)]
    log["nifti_duplicate_keys"] = dup[["accession", "ct_index", "nifti_path", "file_bytes"]].to_dict("records")
    files = files.sort_values(["accession", "ct_index", "file_bytes"], ascending=[True, True, False])
    files = files.drop_duplicates(["accession", "ct_index"], keep="first")

    with ProcessPoolExecutor(cfg["preprocess"]["workers"]) as ex:
        hdr = pd.DataFrame(list(ex.map(header_info, files["nifti_path"], chunksize=32)))
    files = files.merge(hdr, on="nifti_path", how="left")
    files.to_csv(out_dir / "all_nifti_headers.csv", index=False)

    keys = df[[ACC, IDX]].rename(columns={ACC: "accession", IDX: "ct_index"})
    j = keys.merge(files, on=["accession", "ct_index"], how="outer", indicator=True)
    log["label_rows_without_file"] = int((j["_merge"] == "left_only").sum())
    log["files_without_label_row"] = int((j["_merge"] == "right_only").sum())
    unl = j[j["_merge"] == "right_only"]
    log["unlabelled_files_n_slices_in_20_60"] = int(unl["n_slices"].between(20, 60).sum())
    log["unlabelled_files_patient_in_labels"] = int(unl["accession"].isin(keys["accession"]).sum())
    j[j["_merge"] == "left_only"][["accession", "ct_index"]].to_csv(out_dir / "label_rows_without_file.csv", index=False)
    unl[["accession", "ct_index", "nifti_path", "n_slices"]].to_csv(out_dir / "files_without_label.csv", index=False)

    keep_cols = [ACC, IDX, "Radiologists", "Marshall CT score", "The current Brain CT is .......",
                 "Has_Surgery", "is_pre_surgery", "later_operated", "study_datetime", cfg["aux_target"],
                 *classes, *[c for c in df.columns if re.search(r"(EDH|SDH|Contusion) (side|site)_", c)],
                 "Midline Shift", "Has_Mass effect", "Has_Edema",
                 # multi-task / Marshall targets ("Midline shift" is the value in mm, "Midline Shift" the flag)
                 "Midline shift", "Cisterns", "Herniation", "Hydrocephalus", "Has_Skull Fx"]
    m = df[keep_cols].rename(columns={ACC: "accession", IDX: "ct_index", "Midline shift": "midline_shift_mm"}).merge(
        files, on=["accession", "ct_index"], how="left")
    pat_meta = pat[[ACC, "Age", "Sex"]].rename(columns={ACC: "accession"})
    m = m.merge(pat_meta, on="accession", how="left")
    m["has_file"] = m["nifti_path"].notna()
    m["n_slices_in_range"] = m["n_slices"].between(20, 60)
    log["labelled_files_n_slices_out_of_20_60"] = int((m["has_file"] & ~m["n_slices_in_range"]).sum())
    log["labelled_files_header_errors"] = int((m["has_file"] & (m["header_ok"] != True)).sum())  # noqa: E712
    m["usable"] = m["is_pre_surgery"] & m["has_file"] & (m["header_ok"] == True)  # noqa: E712
    m.to_csv(p(cfg, "manifest"), index=False)

    pre = df[df["is_pre_surgery"]]
    log["pre_surgery_stats"] = label_stats(pre, classes)
    log["pre_surgery_vs_spec"] = {c: [log["pre_surgery_stats"]["positives"][c], SPEC_PRE[c]] for c in classes}
    usable = m[m["usable"]]
    log["usable_scans"] = int(len(usable))
    log["usable_patients"] = int(usable["accession"].nunique())
    log["usable_stats"] = label_stats(usable.rename(columns={"accession": ACC, "ct_index": IDX}), classes)
    log["slice_thickness_counts"] = {str(k): int(v) for k, v in
                                     usable["slice_thickness"].round(2).value_counts().head(15).items()}
    log["n_slices_describe"] = usable["n_slices"].describe().round(2).to_dict()
    log["axcodes_counts"] = {str(k): int(v) for k, v in usable["axcodes"].value_counts().items()}
    log["usable_affine_inconsistent"] = int((~usable["affine_consistent"].astype(bool)).sum())
    log["current_ct_is_0_in_usable"] = int((usable["The current Brain CT is ......."] == 0).sum())

    (out_dir / "manifest_report.json").write_text(json.dumps(log, indent=2, default=str))
    short = {k: v for k, v in log.items() if not isinstance(v, (dict, list)) or k.endswith("vs_spec")}
    print(json.dumps(short, indent=2, default=str))


if __name__ == "__main__":
    main()
