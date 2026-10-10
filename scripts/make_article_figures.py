"""Figures for the project article (docs/figures/article/). Reads only saved reports / predictions; no scan keys or
patient images are written except de-identified crops made elsewhere.

  python scripts/make_article_figures.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("ICH_OUTPUTS", "D:/All parts NIFTI files/ich_pipeline/outputs"))
FIG = ROOT / "docs" / "figures" / "article"
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#1f2a33", "#6b7882", "#d9dfe3"
C_MODELS = ["#8a9aa6", "#b9a27a", "#3f7f8c", "#c4553b", "#6c5a8e"]
CLASSES = ["EDH", "SDH", "tSAH", "Has_IVH", "Contusion"]
CLS_LABEL = {"Has_IVH": "IVH"}
RUNS = [("e2e_vh\n(original)", ROOT / "e2e_vh/eval_report.json"),
        ("e2e_vh weights\nnew slices", OUT_ROOT / "runs/e2e_vh_oldbags_newtrim/eval_report.json"),
        ("e2e_vh_ft", ROOT / "e2e_vh_ft/eval_report.json"),
        ("e2e_vh_mt\n(multi-task)", ROOT / "e2e_vh_mt/eval_report.json"),
        ("VisionHOPE-small\n(final)", ROOT / "vh_small_final/eval_report.json")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def reports():
    out = []
    for name, f in RUNS:
        if Path(f).exists():
            out.append((name, json.load(open(f))["splits"]["test"]))
    return out


def fig_macro():
    rs = reports()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, met, title in zip(axes, ("auroc", "auprc"), ("Macro AUROC (test)", "Macro AUPRC (test)")):
        for i, (name, r) in enumerate(rs):
            v, (lo, hi) = r["metrics"]["macro"][met], r["ci95"]["macro"][met]
            ax.bar(i, v, color=C_MODELS[i], width=0.62)
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=4, lw=1)
            ax.text(i, hi + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(rs)), [n for n, _ in rs], fontsize=7.5)
        ax.set_ylim(0.6 if met == "auprc" else 0.75, 0.84 if met == "auprc" else 0.92)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    save(fig, "macro_models.png")


def fig_per_class():
    rs = reports()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    w = 0.8 / len(rs)
    for ax, met in zip(axes, ("auroc", "auprc")):
        for i, (name, r) in enumerate(rs):
            vals = [r["metrics"][c][met] for c in CLASSES]
            ax.bar(np.arange(5) + (i - (len(rs) - 1) / 2) * w, vals, w, color=C_MODELS[i], label=name.replace("\n", " "))
        ax.set_xticks(range(5), [CLS_LABEL.get(c, c) for c in CLASSES])
        ax.set_ylim(0.5 if met == "auprc" else 0.75, 0.95 if met == "auprc" else 0.94)
        ax.set_title(f"Per-class {met.upper()} (test)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, "per_class_models.png")


def fig_tasks():
    r0 = json.load(open(ROOT / "e2e_vh_mt/marshall_report.json"))["tasks"]["test"]
    r = json.load(open(ROOT / "vh_small_final/marshall_report.json"))["tasks"]["test"]
    label = {"mls_any": "midline shift > 0", "mls_gt5": "midline shift > 5 mm", "cist_compressed": "cisterns compressed",
             "cist_effaced": "cisterns effaced", "marshall_ge2": "any pathology (Marshall >= II)",
             "marshall_vi": "Marshall VI (large lesion)", "mass_effect": "mass effect", "herniation": "herniation",
             "edema": "edema", "hydrocephalus": "hydrocephalus", "skull_fx": "skull fracture",
             "sdh_right": "SDH right", "sdh_left": "SDH left", "edh_right": "EDH right", "edh_left": "EDH left",
             "contusion_right": "contusion right", "contusion_left": "contusion left"}
    items = sorted(r.items(), key=lambda kv: kv[1]["auroc"])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, (k, v) in enumerate(items):
        marshall = k in ("mls_gt5", "cist_compressed", "marshall_ge2", "marshall_vi")
        col = "#c4553b" if marshall else "#3f7f8c"
        lo, hi = v["auroc_ci"]
        ax.plot([lo, hi], [i, i], color=col, lw=2, alpha=0.45)
        ax.plot(r0[k]["auroc"], i, "o", mfc="white", mec=MUTED, ms=5)
        ax.plot(v["auroc"], i, "o", color=col, ms=6)
        ax.text(hi + 0.006, i, f"{v['auroc']:.2f}  (prev. {100 * v['prevalence']:.0f} %)", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(items)), [label[k] for k, _ in items])
    ax.axvline(0.5, color=GRID, lw=1)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("AUROC with 95 % CI (test)")
    ax.set_title("Task heads, VisionHOPE-small (filled, 95 % CI) vs tiny multi-task (open)\nred = Marshall components")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "task_heads.png")


def fig_marshall():
    rows = []
    mr = json.load(open(ROOT / "e2e_vh_mt/marshall_report.json"))["marshall"]["test"]
    rows.append(("tiny: rule, per-head F1 thresholds", mr["model_per_head_f1_thresholds"]))
    rows.append(("tiny: rule, thresholds tuned on val", mr["model"]))
    st = json.load(open(ROOT / "e2e_vh_mt/marshall_stack_report.json"))["test"]
    rows.append(("tiny: learned combiner", st))
    fin = json.load(open(ROOT / "e2e_vh_mt/marshall_final_report.json"))["test"]
    rows.append(("tiny: rule + cistern model + volume", fin["rule"]))
    cal = OUT_ROOT / "runs/e2e_vh_mt/calibration.json"
    if cal.exists():
        c = json.load(open(cal))["validation"]["test_rule"]
        rows.append(("tiny: rule + volume", dict(kappa_quadratic=c["kappa_quadratic"], kappa_quadratic_ci95=c["kappa_ci95"],
                                                      accuracy=c["accuracy"])))
    cs = ROOT / "vh_small_final/calibration.json"
    if cs.exists():
        c = json.load(open(cs))["validation"]["test_rule"]
        rows.append(("VisionHOPE-small rule + volume (deployed)",
                     dict(kappa_quadratic=c["kappa_quadratic"], kappa_quadratic_ci95=c["kappa_ci95"], accuracy=c["accuracy"])))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, (name, m) in enumerate(rows):
        k, (lo, hi) = m["kappa_quadratic"], m["kappa_quadratic_ci95"]
        ax.plot([lo, hi], [i, i], color="#3f7f8c", lw=2, alpha=0.45)
        ax.plot(k, i, "o", color="#3f7f8c", ms=7)
        ax.text(hi + 0.008, i, f"kappa {k:.2f}, accuracy {m['accuracy']:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    ax.set_xlim(0.3, 0.8)
    ax.set_xlabel("quadratic-weighted kappa vs radiologist Marshall score (test, 95 % CI)")
    ax.axvline(0.994, color=GRID)
    ax.set_title("Marshall CT score from model outputs")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "marshall_methods.png")


def fig_mls():
    z = np.load(ROOT / "vh_small_final/preds_test.npz", allow_pickle=True)
    y, yh = z["reg_y"][:, 0] * 10, np.clip(z["reg_pred"][:, 0] * 10, 0, None)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    rng = np.random.default_rng(0)
    ax.scatter(y + rng.uniform(-0.25, 0.25, len(y)), yh, s=6, alpha=0.35, color="#3f7f8c", lw=0)
    ax.plot([0, 25], [0, 25], color=MUTED, lw=1, ls="--")
    ax.axvline(5, color=GRID)
    ax.axhline(5, color=GRID)
    ax.set_xlabel("radiologist midline shift (mm)")
    ax.set_ylabel("predicted (mm)")
    ax.set_title(f"Midline shift, VisionHOPE-small (test, r = {np.corrcoef(y, yh)[0, 1]:.2f})")
    ax = axes[1]
    bins = [(0, 0), (0.01, 5), (5.01, 10), (10.01, 40)]
    names = ["0", "0-5", "5-10", "> 10"]
    data = [yh[(y >= a) & (y <= b)] for a, b in bins]
    bp = ax.boxplot(data, tick_labels=names, showfliers=False, patch_artist=True, widths=0.55)
    for b in bp["boxes"]:
        b.set(facecolor="#d8e8ea", edgecolor="#3f7f8c")
    for med in bp["medians"]:
        med.set(color="#c4553b", lw=1.5)
    for i, (a, b) in enumerate(bins, 1):
        ax.text(i, ax.get_ylim()[1] * 0.95, f"n={len(data[i - 1])}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xlabel("radiologist shift group (mm)")
    ax.set_ylabel("predicted (mm)")
    ax.set_title("Predicted shift by true group")
    save(fig, "midline_shift.png")


def fig_volume():
    v = pd.read_csv(OUT_ROOT / "lesion_volume/volumes_test.csv")
    m = pd.read_csv(OUT_ROOT / "scan_manifest.csv")
    m["key"] = m["accession"].astype(int).astype(str) + "_" + m["ct_index"].astype(int).astype(str)
    d = v.merge(m[["key", "Marshall CT score"]], on="key")
    d = d[d["Marshall CT score"] != 5]
    cats = [1, 2, 3, 4, 6]
    data = [np.log1p(d.loc[d["Marshall CT score"] == c, "largest_cm3"]) for c in cats]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bp = ax.boxplot(data, tick_labels=["I", "II", "III", "IV", "VI"], showfliers=False, patch_artist=True, widths=0.55)
    for b in bp["boxes"]:
        b.set(facecolor="#f0dcd5", edgecolor="#c4553b")
    for med in bp["medians"]:
        med.set(color=INK, lw=1.5)
    ticks = [0, 1, 5, 10, 25, 50, 100]
    ax.set_yticks(np.log1p(ticks), [str(t) for t in ticks])
    ax.axhline(np.log1p(25), color="#3f7f8c", ls="--", lw=1)
    ax.text(5.35, np.log1p(25), "25 cm3", color="#3f7f8c", fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("radiologist Marshall category")
    ax.set_ylabel("largest hyperdense lesion (cm3)")
    ax.set_title("HU lesion volume by Marshall category (test)")
    save(fig, "lesion_volume.png")


def fig_curves():
    runs = [("e2e_vh", ROOT / "e2e_vh/history.json", "#8a9aa6"), ("e2e_vh_ft", ROOT / "e2e_vh_ft/history.json", "#3f7f8c"),
            ("e2e_vh_mt", ROOT / "e2e_vh_mt/history.json", "#c4553b")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    off = 0
    for name, f, col in runs:
        if not f.exists():
            continue
        h = json.load(open(f))["history"]
        ep = np.arange(1, len(h["val_auprc"]) + 1) + off
        axes[0].plot(ep, h["train_loss"], color=col, marker=".", label=name)
        axes[1].plot(ep, h["val_auprc"], color=col, marker=".", label=name)
        axes[2].plot(ep, h["train_auprc"], color=col, marker=".", label=name)
        if "val_task_auprc" in h:
            axes[1].plot(ep, h["val_task_auprc"], color=col, ls="--", marker=".", label=f"{name} task heads")
        off = ep[-1]
    for ax, t in zip(axes, ("train loss", "val macro AUPRC (dashed: task heads)", "train macro AUPRC")):
        ax.set_title(t)
        ax.set_xlabel("cumulative Stage B epoch")
        ax.grid(color=GRID, lw=0.6)
    axes[1].legend(fontsize=7, frameon=False)
    save(fig, "training_curves_all.png")
    h1 = json.load(open(ROOT / "vh_small_final/history_epochs1-6.json"))["history"]
    h2 = json.load(open(ROOT / "vh_small_final/history_epochs7-9.json"))["history"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for key, ax, t in (("train_loss", axes[0], "train / val loss"), ("val_auprc", axes[1], "val AUPRC: classes (solid), task heads (dashed)"),
                       ("val_auc", axes[2], "val macro AUROC, 5 types")):
        v = h1[key] + h2[key]
        ep = np.arange(2, len(v) + 2)            # epoch 1 = frozen-encoder warm-up (separate run)
        ax.plot(ep, v, color="#6c5a8e", marker=".", label=key.replace("_", " "))
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(color=GRID, lw=0.6)
    axes[0].plot(np.arange(2, len(h1["val_loss"] + h2["val_loss"]) + 2), h1["val_loss"] + h2["val_loss"], color="#c4553b",
                 marker=".", label="val loss")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].plot(np.arange(2, len(h1["val_task_auprc"] + h2["val_task_auprc"]) + 2), h1["val_task_auprc"] + h2["val_task_auprc"],
                 color="#6c5a8e", ls="--", marker=".")
    axes[2].axhline(0.838, color=MUTED, ls=":", lw=1)
    axes[2].text(9.8, 0.836, "best tiny model (val)", fontsize=7, color=MUTED, ha="right", va="top")
    save(fig, "training_curves_small.png")


def fig_srnl_drift():
    import torch
    pairs = [("tiny: e2e_vh -> e2e_vh_ft", ROOT / "e2e_vh/best.pt", ROOT / "e2e_vh_ft/best.pt"),
             ("tiny: e2e_vh_ft -> e2e_vh_mt", ROOT / "e2e_vh_ft/best.pt", ROOT / "e2e_vh_mt/best.pt"),
             ("small: ImageNet -> final (SRNL lr x5)", OUT_ROOT / "runs/vh_small_all_e1/best.pt", ROOT / "vh_small_final/best.pt")]
    groups = ["encoder SRNL memories", "encoder other", "MIL head"]
    res = {}
    for name, fa, fb in pairs:
        a = torch.load(fa, map_location="cpu", weights_only=False)["model"]
        b = torch.load(fb, map_location="cpu", weights_only=False)["model"]
        acc = {g: [0.0, 0.0] for g in groups}
        for k, va in a.items():
            if k not in b or not torch.is_floating_point(va) or "running_" in k or b[k].shape != va.shape:
                continue
            g = "MIL head" if k.startswith("head.") else ("encoder SRNL memories" if ".srnl." in k else "encoder other")
            acc[g][0] += float((b[k].float() - va.float()).norm()) ** 2
            acc[g][1] += float(va.float().norm()) ** 2
        res[name] = [100 * (acc[g][0] ** 0.5) / max(acc[g][1] ** 0.5, 1e-12) for g in groups]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w = 0.27
    for i, (name, vals) in enumerate(res.items()):
        ax.bar(np.arange(3) + (i - 1) * w, vals, w, color=["#b9a27a", "#3f7f8c", "#6c5a8e"][i], label=name)
        for j, v in enumerate(vals):
            ax.text(j + (i - 1) * w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3), groups)
    ax.set_yscale("log")
    ax.set_ylabel("relative weight change (%)")
    ax.set_title("How much each part moved during fine-tuning")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "srnl_drift.png")
    (FIG / "srnl_drift.json").write_text(json.dumps(res, indent=1))


def fig_marshall_small():
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.calibrate import vi_prob_from
    from sklearn.metrics import confusion_matrix
    cal = json.load(open(ROOT / "vh_small_final/calibration.json"))
    z = np.load(ROOT / "vh_small_final/preds_test.npz", allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    m = pd.read_csv(OUT_ROOT / "scan_manifest.csv")
    m["key"] = m["accession"].astype(int).astype(str) + "_" + m["ct_index"].astype(int).astype(str)
    truth = m.set_index("key").loc[keys, "Marshall CT score"].values.astype(int)
    vols = pd.read_csv(OUT_ROOT / "lesion_volume/volumes_test.csv").set_index("key").loc[
        keys, ["largest_cm3", "total_cm3", "extra_axial_cm3", "intra_axial_cm3"]].fillna(0).values
    ti = {k: i for i, k in enumerate(cal["tasks"])}
    P = z["task_prob"]
    comp = {"marshall_vi": vi_prob_from(cal, P[:, ti["marshall_vi"]], vols), "mls_gt5": P[:, ti["mls_gt5"]],
            "cist_compressed": P[:, ti["cist_compressed"]], "marshall_ge2": P[:, ti["marshall_ge2"]]}
    pred = np.ones(len(keys), int)
    for head, cat in reversed(cal["rule_order"]):
        pred[comp[head] >= cal["rule_thresholds"][head]] = cat
    keep = truth != 5
    cm = confusion_matrix(truth[keep], pred[keep], labels=[1, 2, 3, 4, 6])
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = ["I", "II", "III", "IV", "VI"]
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(5), names)
    ax.set_yticks(range(5), names)
    ax.set_xlabel("predicted (deployed rule)")
    ax.set_ylabel("radiologist Marshall")
    ax.set_title("VisionHOPE-small + lesion volume, test", fontsize=9)
    save(fig, "marshall_confusion_small.png")


if __name__ == "__main__":
    for f in (fig_macro, fig_per_class, fig_tasks, fig_marshall, fig_mls, fig_volume, fig_curves, fig_srnl_drift,
              fig_marshall_small):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print("FAILED", f.__name__, repr(e))
