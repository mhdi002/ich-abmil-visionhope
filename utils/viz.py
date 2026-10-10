"""Plotting for the ICH ABMIL pipeline.

Headless (matplotlib Agg). Every function saves a PNG, optionally logs it to a
TensorBoard ``writer``, and returns the saved path. Classes with no positives or
no negatives in a split are skipped in ROC/PR plots.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)


def _save(fig, out_path, writer=None, tag=None, step=0):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    if writer is not None:
        writer.add_figure(tag or out_path.stem, fig, global_step=step)
    plt.close(fig)
    return str(out_path)


def _valid(y):
    return 0 < y.sum() < len(y)


def _to_display(img):
    """[3,H,W] or [H,W,3] or [H,W] in [0,1] or uint8 -> [H,W,3] float."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------- dataset plots

def plot_class_distribution(label_dict, class_names, out_path, writer=None):
    """label_dict: {split_name: [n_scans, n_classes] binary array}."""
    splits = list(label_dict)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.8 / max(len(splits), 1)
    x = np.arange(len(class_names))
    for i, s in enumerate(splits):
        y = np.asarray(label_dict[s])
        axes[0].bar(x + i * width, y.sum(0), width, label=f"{s} (n={len(y)})")
        axes[1].bar(x + i * width, 100 * y.mean(0), width, label=s)
    for ax, title in zip(axes, ["Positive scans", "Prevalence (%)"]):
        ax.set_xticks(x + width * (len(splits) - 1) / 2)
        ax.set_xticklabels(class_names)
        ax.set_title(title)
        ax.legend(fontsize=8)
    return _save(fig, out_path, writer, "dataset/class_distribution")


def plot_label_cooccurrence(labels, class_names, out_path, writer=None):
    y = np.asarray(labels).astype(int)
    co = y.T @ y
    cond = co / np.maximum(np.diag(co)[:, None], 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title, fmt in [(axes[0], co, "Co-occurrence counts", "d"),
                                (axes[1], cond, "P(col | row)", ".2f")]:
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color="w", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out_path, writer, "dataset/label_cooccurrence")


def plot_bag_statistics(n_slices, thickness, mask_area, out_path, writer=None, kept_slices=None):
    panels = [("Slices per scan (raw)", n_slices), ("Slice thickness (mm)", thickness),
              ("Mean intracranial area per kept slice (cm^2)", mask_area)]
    if kept_slices is not None:
        panels.insert(1, ("Slices per scan (kept)", kept_slices))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for ax, (title, v) in zip(axes, panels):
        v = np.asarray([x for x in v if x is not None and np.isfinite(x)])
        if len(v):
            ax.hist(v, bins=40, color="steelblue")
            q1, q99 = np.percentile(v, [1, 99])
            ax.axvline(q1, color="r", ls="--", lw=1)
            ax.axvline(q99, color="r", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
    return _save(fig, out_path, writer, "dataset/bag_statistics")


def plot_hu_histograms(hist_pos, hist_neg, bin_edges, class_names, out_path, writer=None):
    """hist_pos/neg: {class: [n_bins] mean normalized histogram over intracranial voxels}."""
    n = len(class_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=True)
    centers = 0.5 * (np.asarray(bin_edges[:-1]) + np.asarray(bin_edges[1:]))
    for ax, c in zip(np.atleast_1d(axes), class_names):
        ax.plot(centers, hist_neg[c], label="negative", color="gray")
        ax.plot(centers, hist_pos[c], label="positive", color="crimson")
        ax.axvspan(50, 80, color="orange", alpha=0.15)
        ax.set_title(c)
        ax.set_xlabel("HU")
        ax.set_yscale("log")
    np.atleast_1d(axes)[0].legend()
    return _save(fig, out_path, writer, "dataset/hu_histograms")


def plot_scan_montage(slices, out_path, title="", writer=None, max_slices=40, channel=0):
    """slices: [N,3,H,W] (uint8 or float). Orientation QC: frontal lobe must be at the top."""
    slices = np.asarray(slices)[:max_slices]
    n = len(slices)
    cols = min(10, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows + 0.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, s in enumerate(slices):
        axes[i].imshow(_to_display(s)[..., channel] if _to_display(s).ndim == 3 else _to_display(s),
                       cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(str(i), fontsize=6)
    fig.suptitle(title + "  (anterior should be UP)", fontsize=9)
    return _save(fig, out_path, writer, "dataset/scan_montage")


def plot_preprocessing_check(raw_hu, mask, windows, out_path, title="", writer=None):
    """raw_hu [H,W], mask [H,W], windows [3,H,W] in [0,1]."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    axes[0].imshow(np.clip(raw_hu, -200, 1500), cmap="gray")
    axes[0].set_title("raw HU [-200,1500]")
    axes[1].imshow(np.clip(raw_hu, 0, 100), cmap="gray")
    axes[1].contour(mask, levels=[0.5], colors="lime", linewidths=0.8)
    axes[1].set_title("mask outline")
    for k, name in enumerate(["brain L40/W80", "subdural L80/W200", "stroke L40/W40"]):
        axes[2 + k].imshow(windows[k], cmap="gray", vmin=0, vmax=1)
        axes[2 + k].set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    return _save(fig, out_path, writer, "dataset/preprocessing_check")


def plot_positive_examples_per_class(examples, class_names, out_path, writer=None, channel=0):
    """examples: {class: list of (slice [3,H,W], caption)}; one row per class."""
    cols = max((len(examples.get(c, [])) for c in class_names), default=1) or 1
    fig, axes = plt.subplots(len(class_names), cols, figsize=(1.9 * cols, 2.1 * len(class_names)))
    axes = np.atleast_2d(axes)
    for r, c in enumerate(class_names):
        for k in range(cols):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(examples.get(c, [])):
                img, cap = examples[c][k]
                d = _to_display(img)
                ax.imshow(d[..., channel] if d.ndim == 3 else d, cmap="gray", vmin=0, vmax=1)
                ax.set_title(cap, fontsize=6)
        axes[r, 0].text(-0.1, 0.5, c, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=10)
    return _save(fig, out_path, writer, "dataset/positive_examples")


def plot_augmentation_preview(original, augmented_list, out_path, writer=None, channel=0):
    n = 1 + len(augmented_list)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    for ax, img, t in zip(axes, [original, *augmented_list], ["original"] + [f"aug {i}" for i in range(1, n)]):
        d = _to_display(img)
        ax.imshow(d[..., channel] if d.ndim == 3 else d, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    return _save(fig, out_path, writer, "dataset/augmentation_preview")


# ---------------------------------------------------------------- training plots

def save_epoch_plots(history, per_class_auc, class_names, out_dir, writer=None, epoch=0, per_class_recall=None):
    """history keys: train_loss, val_loss, train_auc, val_auc, train_auprc, val_auprc,
    lr, attn_entropy, attn_peak, grad_norm, plus thresholded (0.5) train_acc, val_acc, val_bal_acc,
    train_recall, val_recall, val_precision, val_specificity (lists, one value per epoch).
    per_class_auc / per_class_recall: {class: list of val values per epoch}."""
    out_dir = Path(out_dir)
    ep = np.arange(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    pairs = [(axes[0, 0], "Loss", ("train_loss", "val_loss")), (axes[0, 1], "Macro AUROC", ("train_auc", "val_auc")),
             (axes[0, 2], "Macro AUPRC", ("train_auprc", "val_auprc", "val_task_auprc")),
             (axes[2, 0], "Macro accuracy @0.5", ("train_acc", "val_acc", "val_bal_acc")),
             (axes[2, 1], "Macro recall (sensitivity) @0.5", ("train_recall", "val_recall")),
             (axes[2, 2], "Val precision / specificity @0.5", ("val_precision", "val_specificity"))]
    for ax, t, keys in pairs:
        for k in keys:
            if history.get(k):
                ax.plot(ep[:len(history[k])], history[k], marker=".", label=k)
        ax.set_title(t)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax = axes[2, 3]
    for c in class_names:
        v = (per_class_recall or {}).get(c, [])
        if v:
            ax.plot(np.arange(1, len(v) + 1), v, marker=".", label=c)
    ax.set_title("Per-class val recall @0.5")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    singles = [(axes[0, 3], "lr", "Learning rate"), (axes[1, 0], "attn_entropy", "Attention entropy (normalized)"),
               (axes[1, 1], "attn_peak", "Mean attention peak"), (axes[1, 2], "grad_norm", "Gradient norm")]
    for ax, k, t in singles:
        if history.get(k):
            ax.plot(ep[:len(history[k])], history[k], marker=".")
        ax.set_title(t)
        ax.grid(alpha=0.3)
    ax = axes[1, 3]
    for c in class_names:
        v = per_class_auc.get(c, [])
        if v:
            ax.plot(np.arange(1, len(v) + 1), v, marker=".", label=c)
    ax.set_title("Per-class val AUROC")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    if writer is not None:
        for k, v in history.items():
            if v:
                writer.add_scalar(f"epoch/{k}", v[-1], epoch)
        for c, v in per_class_auc.items():
            if v:
                writer.add_scalar(f"val_auc/{c}", v[-1], epoch)
        for c, v in (per_class_recall or {}).items():
            if v:
                writer.add_scalar(f"val_recall/{c}", v[-1], epoch)
    return _save(fig, out_dir / "training_curves.png", writer, "train/curves", epoch)


def plot_attention_snapshot(scans, class_names, out_path, writer=None, epoch=0, channel=0, topk=3):
    """scans: list of dicts {slices [N,3,H,W], attn [C,N], labels [C], probs [C], name}."""
    n = len(scans)
    fig, axes = plt.subplots(n, 1 + topk, figsize=(3 * (1 + topk), 2.8 * n), squeeze=False)
    for r, s in enumerate(scans):
        attn = np.asarray(s["attn"])
        ax = axes[r, 0]
        for c, name in enumerate(class_names):
            ax.plot(attn[c], label=f"{name} y={int(s['labels'][c])} p={s['probs'][c]:.2f}")
        ax.set_title(s.get("name", ""), fontsize=7)
        ax.legend(fontsize=5)
        ax.set_xlabel("slice")
        pos = [c for c in range(len(class_names)) if s["labels"][c] > 0]
        c_show = pos[0] if pos else int(np.argmax(s["probs"]))
        top = np.argsort(-attn[c_show])[:topk]
        for k, i in enumerate(top):
            d = _to_display(s["slices"][i])
            axes[r, 1 + k].imshow(d[..., channel] if d.ndim == 3 else d, cmap="gray", vmin=0, vmax=1)
            axes[r, 1 + k].set_title(f"{class_names[c_show]} slice {i} a={attn[c_show, i]:.2f}", fontsize=7)
            axes[r, 1 + k].axis("off")
    fig.suptitle(f"Attention snapshot epoch {epoch}")
    return _save(fig, out_path, writer, "train/attention_snapshot", epoch)


# ---------------------------------------------------------------- evaluation plots

def plot_roc_pr(labels, probs, class_names, out_path, tag="", writer=None):
    labels, probs = np.asarray(labels), np.asarray(probs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c, name in enumerate(class_names):
        y, s = labels[:, c], probs[:, c]
        if not _valid(y):
            continue
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, label=f"{name} {auc(fpr, tpr):.3f}")
        pr, rc, _ = precision_recall_curve(y, s)
        axes[1].plot(rc, pr, label=f"{name} {average_precision_score(y, s):.3f} (prev {y.mean():.2f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title=f"ROC {tag}", xlabel="FPR", ylabel="TPR")
    axes[1].set(title=f"PR {tag}", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    return _save(fig, out_path, writer, f"eval/{tag}_roc_pr")


def plot_calibration(labels, probs, class_names, out_path, tag="", writer=None, n_bins=10):
    labels, probs = np.asarray(labels), np.asarray(probs)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for c, name in enumerate(class_names):
        if not _valid(labels[:, c]):
            continue
        frac, mean = calibration_curve(labels[:, c], probs[:, c], n_bins=n_bins, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=3, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title=f"Calibration {tag}", xlabel="Mean predicted", ylabel="Fraction positive")
    ax.legend(fontsize=8)
    return _save(fig, out_path, writer, f"eval/{tag}_calibration")


def plot_score_distributions(labels, probs, class_names, out_path, tag="", writer=None):
    labels, probs = np.asarray(labels), np.asarray(probs)
    n = len(class_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.3))
    for ax, c, name in zip(np.atleast_1d(axes), range(n), class_names):
        y = labels[:, c]
        ax.hist(probs[y == 0, c], bins=30, alpha=0.6, density=True, label="neg", color="gray")
        ax.hist(probs[y == 1, c], bins=30, alpha=0.6, density=True, label="pos", color="crimson")
        ax.set_title(name)
    np.atleast_1d(axes)[0].legend()
    fig.suptitle(f"Score distributions {tag}")
    return _save(fig, out_path, writer, f"eval/{tag}_scores")


def plot_confusion_matrices(labels, probs, thresholds, class_names, out_path, tag="", writer=None):
    labels, probs = np.asarray(labels), np.asarray(probs)
    n = len(class_names)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    for ax, c, name in zip(np.atleast_1d(axes), range(n), class_names):
        y, pred = labels[:, c].astype(int), (probs[:, c] >= thresholds[c]).astype(int)
        cm = np.array([[((y == a) & (pred == b)).sum() for b in (0, 1)] for a in (0, 1)])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set(xlabel="pred", ylabel="true", title=f"{name} t={thresholds[c]:.2f}")
    fig.suptitle(f"Confusion at val-selected thresholds ({tag})")
    return _save(fig, out_path, writer, f"eval/{tag}_confusion")


def save_full_eval_plots(labels, probs, thresholds, class_names, out_dir, tag="val", writer=None):
    out_dir = Path(out_dir)
    return [
        plot_roc_pr(labels, probs, class_names, out_dir / f"{tag}_roc_pr.png", tag, writer),
        plot_calibration(labels, probs, class_names, out_dir / f"{tag}_calibration.png", tag, writer),
        plot_score_distributions(labels, probs, class_names, out_dir / f"{tag}_scores.png", tag, writer),
        plot_confusion_matrices(labels, probs, thresholds, class_names, out_dir / f"{tag}_confusion.png", tag, writer),
    ]


def plot_metric_bars_with_ci(metrics, class_names, out_path, tag="", writer=None):
    """metrics: {metric_name: {class: (point, lo, hi)}}."""
    names = list(metrics)
    ncol = min(3, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, m in zip(axes.flat, names):
        cls = [c for c in class_names + ["macro"] if c in metrics[m]]
        pt = np.array([metrics[m][c][0] for c in cls])
        lo = np.array([metrics[m][c][1] for c in cls])
        hi = np.array([metrics[m][c][2] for c in cls])
        ax.bar(range(len(cls)), pt, yerr=[pt - lo, hi - pt], capsize=4, color="steelblue")
        for i, v in enumerate(pt):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(cls)))
        ax.set_xticklabels(cls, rotation=30)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{m} {tag} (95% bootstrap CI)")
    return _save(fig, out_path, writer, f"eval/{tag}_metric_bars")


def plot_attention_gallery(items, class_name, out_path, writer=None, channel=0):
    """items: {'TP'|'FP'|'FN': list of dicts {slices [k,3,H,W], attn [k], prob, name}}."""
    groups = [g for g in ("TP", "FP", "FN") if items.get(g)]
    rows = sum(len(items[g]) for g in groups)
    if rows == 0:
        return None
    k = max(len(it["slices"]) for g in groups for it in items[g])
    fig, axes = plt.subplots(rows, k, figsize=(2.2 * k, 2.3 * rows), squeeze=False)
    r = 0
    for g in groups:
        for it in items[g]:
            for j in range(k):
                ax = axes[r, j]
                ax.axis("off")
                if j < len(it["slices"]):
                    d = _to_display(it["slices"][j])
                    ax.imshow(d[..., channel] if d.ndim == 3 else d, cmap="gray", vmin=0, vmax=1)
                    if "cam" in it and it["cam"] is not None:
                        ax.imshow(it["cam"][j], cmap="jet", alpha=0.35, vmin=0, vmax=1)
                    ax.set_title(f"{g} {it['name']} p={it['prob']:.2f} a={it['attn'][j]:.2f}", fontsize=6)
            r += 1
    fig.suptitle(f"Top-attention slices: {class_name}")
    return _save(fig, out_path, writer, f"eval/attention_gallery_{class_name}")


def plot_ablation_table(rows, out_path, writer=None, columns=None):
    """rows: list of dicts (one per ablation); rendered as a table image."""
    if not rows:
        return None
    columns = columns or list(rows[0])
    cell = [[(f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, ""))) for c in columns] for r in rows]
    fig, ax = plt.subplots(figsize=(min(2.0 * len(columns), 26), 0.45 * len(rows) + 1))
    ax.axis("off")
    t = ax.table(cellText=cell, colLabels=columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.3)
    return _save(fig, out_path, writer, "eval/ablation_table")


def threshold_metrics(labels, probs, thr=0.5):
    """Per-class and macro accuracy / recall / precision / specificity / balanced accuracy / F1 at threshold(s)."""
    labels, probs = np.asarray(labels).astype(int), np.asarray(probs)
    thr = np.broadcast_to(np.asarray(thr, dtype=float), (labels.shape[1],))
    per = []
    for c in range(labels.shape[1]):
        y, yh = labels[:, c], (probs[:, c] >= thr[c]).astype(int)
        tp, tn = int(((y == 1) & (yh == 1)).sum()), int(((y == 0) & (yh == 0)).sum())
        fp, fn = int(((y == 0) & (yh == 1)).sum()), int(((y == 1) & (yh == 0)).sum())
        rec = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        prec = tp / (tp + fp) if tp + fp else 0.0
        per.append(dict(accuracy=(tp + tn) / len(y), recall=rec, precision=prec, specificity=spec,
                        balanced_accuracy=np.nanmean([rec, spec]),
                        f1=2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0))
    macro = {k: float(np.nanmean([d[k] for d in per])) for k in per[0]}
    return per, macro


def macro_metrics(labels, probs):
    labels, probs = np.asarray(labels), np.asarray(probs)
    aucs, aps = [], []
    for c in range(labels.shape[1]):
        if _valid(labels[:, c]):
            aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
            aps.append(average_precision_score(labels[:, c], probs[:, c]))
    return (float(np.mean(aucs)) if aucs else float("nan"), float(np.mean(aps)) if aps else float("nan"))
