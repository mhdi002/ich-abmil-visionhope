"""Phase 1 preprocessing: NIfTI -> axial slices -> intracranial mask -> 3 CT windows -> uint8 cache.

Stages (run in order):
  python preprocess.py --stage cache    # per-scan cache + stats (parallel)
  python preprocess.py --stage trim     # strip near-empty skull-base / vertex end slices in place
  python preprocess.py --stage orient   # 4-way anterior-direction classifier, fixes cache, review CSV
  python preprocess.py --stage plots    # dataset plots
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from utils.config import load_config, p, parse_overrides, seed_everything


# ---------------------------------------------------------------- volume loading

def load_axial_hu(row):
    """Return HU volume [Z,H,W] in display orientation and (row_mm, col_mm, slice_mm).

    Display orientation: image rows top->bottom = anterior->posterior, columns
    left->right = patient right->left (radiological). Affine-consistent scans use
    the canonical RAS reorientation (pure axis permutation/flip, no resampling).
    Inconsistent affines fall back to the short axis as the slice axis with an
    LAS-like in-plane guess; the orientation stage verifies and corrects both.
    """
    img = nib.load(row["nifti_path"])
    if img.ndim == 4:
        img = img.slicer[..., 0]
    if bool(row["affine_consistent"]):
        img = nib.as_closest_canonical(img)
        vol = img.get_fdata(dtype=np.float32)            # X=R, Y=A, Z=S
        zx, zy, zz = img.header.get_zooms()[:3]
        vol = vol[::-1, ::-1, :]                          # X: R->L, Y: A->P
        vol = np.transpose(vol, (2, 1, 0))                # [Z, Y(rows), X(cols)]
        return np.ascontiguousarray(vol), (float(zy), float(zx), float(zz))
    vol = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    ax = int(row["slice_axis"])
    inplane = [i for i in range(3) if i != ax]
    vol = np.moveaxis(vol, ax, 0)                         # [Z, a0, a1]
    vol = vol[:, :, ::-1]                                 # assume a1 increases anteriorly
    vol = np.transpose(vol, (0, 2, 1))                    # [Z, a1(rows), a0(cols)]
    return np.ascontiguousarray(vol), (float(zooms[inplane[1]]), float(zooms[inplane[0]]), float(zooms[ax]))


def _disk(r):
    r = max(int(round(r)), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _largest_components(mask, keep_frac=0.15):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.flatnonzero(sizes >= keep_frac * sizes.max()) + 1
    return np.isin(lab, keep)


def skull_interior(hu, px_mm, cfg):
    """Region enclosed by a closed skull ring on this slice, or None (skull base, neck, vertex)."""
    pc = cfg["preprocess"]
    body = _largest_components(hu > -400, keep_frac=1.0)
    head = ndi.binary_fill_holes(body)
    skull = (hu > pc["bone_hu"]) & body          # excludes head holders / table arcs
    interior = None
    if skull.sum() * px_mm ** 2 > 200:
        closed = ndi.binary_closing(skull, structure=_disk(3.0 / px_mm), iterations=1)
        filled = ndi.binary_fill_holes(closed) & ~closed
        if filled.sum() * px_mm ** 2 > 500:
            interior = _largest_components(filled, keep_frac=1.0)
    return interior, head, skull


def intracranial_mask(hu, px_mm, cfg, interior, head, skull, ref):
    """Tight intracranial tissue mask for one slice (generous dilation is applied by the caller).

    Slices without a closed skull ring are restricted to ``ref``: the dilated union of the
    closed-ring interiors of the scan, which keeps neck/shoulder/face tissue out.
    """
    pc = cfg["preprocess"]
    soft = (hu > pc["soft_hu"][0]) & (hu < pc["soft_hu"][1]) & head
    if interior is not None:
        cand = soft & interior
    else:
        skull_d = ndi.binary_dilation(skull, structure=_disk(1.0 / px_mm))
        cand = soft & ~skull_d
        if ref is not None:
            cand &= ref
    er = _disk(1.5 / px_mm)
    core = ndi.binary_erosion(cand, structure=er)
    core = _largest_components(core)
    mask = ndi.binary_dilation(core, structure=er) & cand
    return ndi.binary_fill_holes(mask)


def cranial_range(ring, Z, slack=2, max_gap=2):
    """Slice range of the longest run of closed-skull-ring slices (gaps <= max_gap merged), +- slack.

    Excludes neck/face slices below the skull base, where muscle mimics blood density.
    """
    if not ring:
        return 0, Z - 1
    runs, start, prev = [], ring[0], ring[0]
    for z in ring[1:]:
        if z - prev > max_gap + 1:
            runs.append((start, prev))
            start = z
        prev = z
    runs.append((start, prev))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return max(a - slack, 0), min(b + slack, Z - 1)


def edge_trim(out, pc, min_keep=3):
    """Contiguous end-trim of a cached uint8 stack [N,3,H,W] (index 0 = skull base).

    Area is soft-tissue pixels of the brain-window channel (0 < v < 255) relative to the scan's
    largest slice, so the crop scale cancels out. Interior slices are never removed.
    """
    c = out[:, 0]
    area = ((c > 0) & (c < 255)).reshape(len(out), -1).sum(1).astype(np.float64)
    r = area / max(area.max(), 1.0)
    lo, hi = 0, len(out) - 1
    while hi - lo + 1 > min_keep and r[lo] < pc.get("trim_base_frac", 0.0):
        lo += 1
    while hi - lo + 1 > min_keep and r[hi] < pc.get("trim_vertex_frac", 0.0):
        hi -= 1
    return lo, hi


def inner_bone_cm2(hu, mask, px_mm, row_mm, col_mm, bone_hu, margin_mm=2.0):
    """Bone area (cm2) inside the convex hull of the intracranial mask, ``margin_mm`` away from its edge.

    Supratentorial slices: the calvarial ring lies outside the hull, so this is ~0 (small calcifications
    only). Skull-base slices: petrous ridges, clivus, sphenoid and orbital roofs sit between the fossae,
    inside the hull. Independent of head holders / pillows, unlike depth below the head surface.
    """
    if not mask.any():
        return 0.0
    from skimage.morphology import convex_hull_image
    hull = convex_hull_image(mask)
    hull = ndi.distance_transform_edt(hull) * px_mm > margin_mm
    return float(((hu > bone_hu) & hull).sum() * row_mm * col_mm / 100.0)


def anatomic_trim(area_cm2, bone_cm2, pc):
    """Contiguous end-trim of a bag ordered inferior -> superior; returns (lo, hi) inclusive.

    Base end: drop slices whose intracranial area is < trim_base_frac of the scan max, or whose inner
    bone is > trim_base_bone_frac of the max intracranial area (petrous / orbit / sphenoid levels).
    Vertex end: drop slices with area < trim_vertex_frac of max. Interior slices are never removed;
    at least trim_min_keep slices and at most trim_max_frac of the bag are dropped.
    """
    n = len(area_cm2)
    amax = max(float(np.max(area_cm2)), 1e-6)
    r, b = np.asarray(area_cm2) / amax, np.asarray(bone_cm2) / amax
    min_keep = min(n, max(pc.get("trim_min_keep", 3), int(np.ceil(n * (1 - pc.get("trim_max_frac", 1.0))))))
    lo, hi = 0, n - 1
    while hi - lo + 1 > min_keep and (r[lo] < pc["trim_base_frac"] or b[lo] > pc["trim_base_bone_frac"]):
        lo += 1
    while hi - lo + 1 > min_keep and r[hi] < pc["trim_vertex_frac"]:
        hi -= 1
    return lo, hi


def window(hu, level, width):
    lo = level - width / 2.0
    return np.clip((hu - lo) / width, 0.0, 1.0)


def _resize_stack(x, out):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    t = torch.from_numpy(np.ascontiguousarray(x))
    return F.interpolate(t, size=(out, out), mode="bilinear", align_corners=False, antialias=True).numpy()


def scan_masks(hu, px, row_mm, col_mm, cfg):
    """Per-slice tight intracranial masks, head masks, intracranial area (cm2), closed-ring slices, the cranial
    range (lo, hi) and the slices inside it with enough intracranial area (before any end trim)."""
    pc = cfg["preprocess"]
    Z = hu.shape[0]
    parts = [skull_interior(hu[z], px, cfg) for z in range(Z)]
    ring = [z for z in range(Z) if parts[z][0] is not None]

    def nearest_ref(z):
        # interior of the nearest closed-ring slice, slightly dilated (robust to head tilt)
        if not ring or parts[z][0] is not None:
            return None
        zr = min(ring, key=lambda q: abs(q - z))
        return ndi.distance_transform_edt(~parts[zr][0]) <= 5.0 / px

    masks = np.stack([intracranial_mask(hu[z], px, cfg, *parts[z], nearest_ref(z)) for z in range(Z)])
    heads = np.stack([parts[z][1] for z in range(Z)])
    area_cm2 = masks.reshape(Z, -1).sum(1) * row_mm * col_mm / 100.0
    lo, hi = cranial_range(ring, Z, pc.get("ring_slack", 2))
    keep = np.flatnonzero((area_cm2 >= pc["min_mask_area_cm2"]) & (np.arange(Z) >= lo) & (np.arange(Z) <= hi))
    ref_core = np.any([parts[z][0] for z in ring], axis=0) if ring else None
    return dict(masks=masks, heads=heads, area_cm2=area_cm2, ring=ring, lo=lo, hi=hi, keep=keep, ref_core=ref_core)


def process_scan(row, cfg, out_dir, debug=False):
    pc = cfg["preprocess"]
    key = f"{int(row['accession'])}_{int(row['ct_index'])}"
    hu, (row_mm, col_mm, slice_mm) = load_axial_hu(row)
    px = float(np.sqrt(row_mm * col_mm))
    Z = hu.shape[0]
    sm = scan_masks(hu, px, row_mm, col_mm, cfg)
    masks, heads, area_cm2, ring, ref_core = sm["masks"], sm["heads"], sm["area_cm2"], sm["ring"], sm["ref_core"]
    lo, hi, keep = sm["lo"], sm["hi"], sm["keep"]
    stats = dict(key=key, n_raw=int(Z), row_mm=row_mm, col_mm=col_mm, slice_mm=slice_mm,
                 cranial_lo=int(lo), cranial_hi=int(hi))
    if len(keep) < 3:
        stats.update(ok=False, reason=f"only {len(keep)} slices with intracranial area")
        return stats
    keep_pre = keep
    mode = pc.get("trim_mode", "area")
    anatomic = mode in ("anatomic", "cistern")
    if anatomic:
        bone = np.array([inner_bone_cm2(hu[z], masks[z], px, row_mm, col_mm, pc["bone_hu"], pc["trim_hull_margin_mm"])
                         for z in keep])
        t0, t1 = anatomic_trim(area_cm2[keep], bone, pc)
        stats.update(trim_area=[round(float(v), 2) for v in area_cm2[keep]],
                     trim_bone=[round(float(v), 2) for v in bone], anatomic_trim=[int(t0), int(len(keep) - 1 - t1)])
        if mode == "anatomic":
            keep = keep[t0:t1 + 1]
        else:
            # basal-cistern window around the skull-base trim line: the suprasellar / ambient / quadrigeminal
            # cisterns sit just above the petrous ridges and dorsum sellae, the 3rd ventricle above them
            below = int(round(pc["cistern_below_mm"] / slice_mm))
            above = int(round(pc["cistern_above_mm"] / slice_mm))
            a, b = max(0, t0 - below), min(len(keep) - 1, t0 + above)
            stats.update(cistern_window=[int(a), int(b)])
            keep = keep[a:b + 1]

    r = pc["mask_dilation_mm"] / px
    gen = np.stack([(ndi.distance_transform_edt(~masks[z]) <= r) & heads[z] if masks[z].any()
                    else masks[z] for z in keep])
    hu_k, m_k = hu[keep], masks[keep]

    vox = hu_k[m_k]
    edges = np.asarray(pc["hu_hist_bins"], dtype=np.float32)
    hist = np.histogram(np.clip(vox, edges[0], edges[-1] - 1e-3), bins=edges)[0] / max(len(vox), 1)
    vox_cm3 = row_mm * col_mm * slice_mm / 1000.0
    pct = np.percentile(vox, [5, 25, 50, 75, 95])
    stats.update(
        ok=True, n_kept=int(len(keep)), kept_first=int(keep[0]), kept_last=int(keep[-1]),
        mask_area_mean_cm2=float(area_cm2[keep].mean()), mask_area_max_cm2=float(area_cm2[keep].max()),
        intracranial_vol_cm3=float(m_k.sum() * vox_cm3),
        hu_mean=float(vox.mean()), hu_std=float(vox.std()),
        **{f"hu_p{q}": float(v) for q, v in zip([5, 25, 50, 75, 95], pct)},
        frac_50_80=float(((vox >= 50) & (vox <= 80)).mean()),
        frac_60_90=float(((vox >= 60) & (vox <= 90)).mean()),
        vol_50_80_cm3=float(((vox >= 50) & (vox <= 80)).sum() * vox_cm3),
        slice_max_frac_50_80=float(max(((hu_k[i][m_k[i]] >= 50) & (hu_k[i][m_k[i]] <= 80)).mean()
                                       for i in range(len(keep)))),
        **{f"hist_{i}": float(v) for i, v in enumerate(hist)},
    )

    union = gen.any(0)
    if ref_core is not None:
        union = (ndi.distance_transform_edt(~ref_core) <= r) & union
        if not union.any():
            union = gen.any(0)
    stats["n_ring_slices"] = len(ring)
    rows_any, cols_any = np.flatnonzero(union.any(1)), np.flatnonzero(union.any(0))
    cy, cx = (rows_any[0] + rows_any[-1]) / 2.0, (cols_any[0] + cols_any[-1]) / 2.0
    half_mm = max((rows_any[-1] - rows_any[0]) * row_mm, (cols_any[-1] - cols_any[0]) * col_mm) / 2.0
    half_mm *= 1.0 + pc["crop_margin"]
    hr, hc = int(round(half_mm / row_mm)), int(round(half_mm / col_mm))
    H, W = hu.shape[1:]
    pad = max(hr, hc) + 2
    ch = []
    hu_masked = np.where(gen, hu_k, -1024.0).astype(np.float32)
    for level, width in pc["windows"]:
        ch.append(window(hu_masked, level, width))
    x = np.stack(ch, 1)                                                 # [N,3,H,W]
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    y0, x0 = int(round(cy)) + pad - hr, int(round(cx)) + pad - hc
    x = x[:, :, y0:y0 + 2 * hr, x0:x0 + 2 * hc]
    x = _resize_stack(x, pc["cache_size"])
    out = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    stats.update(crop_mm=float(2 * half_mm), crop_center=[float(cy), float(cx)], orig_hw=[int(H), int(W)])
    t0, t1 = (0, len(out) - 1) if anatomic else edge_trim(out, pc)
    out = out[t0:t1 + 1]
    stats.update(edge_trim=[int(t0), int(len(keep) - 1 - t1)], n_kept=int(len(out)),
                 kept_first=int(keep[t0]), kept_last=int(keep[t1]))
    if debug:
        return stats, dict(hu=hu_k, mask=m_k, gen=gen, out=out, keep=keep, keep_pre=keep_pre, hu_all=hu)
    if pc.get("cache_compress"):
        np.savez_compressed(out_dir / f"{key}.npz", x=out)       # background is zero: ~2-3x smaller
    else:
        np.save(out_dir / f"{key}.npy", out)
    (out_dir / f"{key}.json").write_text(json.dumps(stats))
    return stats


def _worker(args):
    row, cfg, out_dir = args
    try:
        return process_scan(row, cfg, Path(out_dir))
    except Exception as e:  # noqa: BLE001 - recorded per scan
        return dict(key=f"{int(row['accession'])}_{int(row['ct_index'])}", ok=False, reason=repr(e)[:300])


def stage_cache(cfg, overwrite=False, limit=None):
    out_dir = p(cfg, "slice_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(p(cfg, "manifest"))
    m = m[m["usable"]].reset_index(drop=True)
    if limit:
        m = m.sample(limit, random_state=cfg["seed"])
    todo = [r for r in m.to_dict("records")
            if overwrite or not (out_dir / f"{int(r['accession'])}_{int(r['ct_index'])}.json").exists()]
    print(f"{len(m)} usable scans, {len(todo)} to process")
    failed = []
    with ProcessPoolExecutor(cfg["preprocess"]["workers"]) as ex:
        futs = [ex.submit(_worker, (r, cfg, str(out_dir))) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            s = f.result()
            if not s.get("ok"):
                failed.append(s)
                (out_dir / f"{s['key']}.json").write_text(json.dumps(s))
            if i % 250 == 0:
                print(f"  {i}/{len(todo)} done, {len(failed)} failed", flush=True)
    collect_stats(cfg)


def stage_trim(cfg):
    """Apply edge_trim to an existing cache in place (idempotent via the 'edge_trim' json field)."""
    out_dir = p(cfg, "slice_cache")
    pc = cfg["preprocess"]
    n_scans = n_base = n_vertex = 0
    for jf in sorted(out_dir.glob("*.json")):
        s = json.loads(jf.read_text())
        if not s.get("ok") or "edge_trim" in s:
            continue
        npy = jf.with_suffix(".npy")
        out = np.load(npy)
        t0, t1 = edge_trim(out, pc)
        b, v = t0, len(out) - 1 - t1
        if b or v:
            np.save(npy, np.ascontiguousarray(out[t0:t1 + 1]))
        first = s["kept_first"]
        s.update(edge_trim=[int(b), int(v)], n_kept=int(t1 - t0 + 1),
                 kept_first=int(first + t0), kept_last=int(s["kept_last"] - v))
        jf.write_text(json.dumps(s))
        n_scans += 1
        n_base += b
        n_vertex += v
    print(f"edge trim: {n_scans} scans, removed {n_base} base + {n_vertex} vertex slices")
    collect_stats(cfg)


def collect_stats(cfg):
    out_dir = p(cfg, "slice_cache")
    rows = [json.loads(f.read_text()) for f in sorted(out_dir.glob("*.json"))]
    df = pd.DataFrame(rows)
    df.to_csv(p(cfg, "outputs") / "preprocess_stats.csv", index=False)
    ok = df[df["ok"] == True]  # noqa: E712
    print(f"stats: {len(df)} scans, {len(ok)} ok, {len(df) - len(ok)} failed")
    if len(df) > len(ok):
        print(df[df["ok"] != True][["key", "reason"]].head(20).to_string())  # noqa: E712
    return df


# ---------------------------------------------------------------- orientation

ROTS = {0: "anterior up", 1: "anterior left", 2: "anterior down", 3: "anterior right"}


def orientation_net():
    """4-way anterior-direction CNN on [3,64,64] thumbnails (stage_orient trains it, predict_scan.py reuses it)."""
    from torch import nn
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(128, 4))


def orientation_thumb(arr, size=64, fracs=(0.25, 0.5, 0.75)):
    """Brain-window thumbnails of 3 representative slices -> [3,size,size] float.

    With the anatomic trim the skull base is gone from the bag, so lower fractions (config
    preprocess.orient_thumb_fracs) keep the asymmetric anterior/posterior anatomy in view.
    """
    import torch
    import torch.nn.functional as F
    n = len(arr)
    idx = [int(round(f * (n - 1))) for f in fracs]
    x = torch.from_numpy(arr[idx, 1].astype(np.float32) / 255.0)[None]   # subdural window shows bone edge
    return F.interpolate(x, size=(size, size), mode="area")[0].numpy()


def stage_orient(cfg):
    """4-way anterior-direction classifier trained on rotated affine-consistent LAS scans.

    Mirror (left-right) cannot be inferred from anatomy and is taken from the affine;
    it is checked separately against laterality labels in eval.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seed_everything(cfg["seed"])
    out_dir = p(cfg, "outputs") / "orientation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = p(cfg, "slice_cache")
    m = pd.read_csv(p(cfg, "manifest"))
    st = pd.read_csv(p(cfg, "outputs") / "preprocess_stats.csv")
    m["key"] = m["accession"].astype(int).astype(str) + "_" + m["ct_index"].astype(int).astype(str)
    m = m[m["usable"]].merge(st[["key", "ok"]], on="key")
    m = m[m["ok"] == True].reset_index(drop=True)  # noqa: E712
    print("loading thumbnails", len(m))
    with ProcessPoolExecutor(16) as ex:
        fr = tuple(cfg["preprocess"].get("orient_thumb_fracs", (0.25, 0.5, 0.75)))
        thumbs = np.stack(list(ex.map(_thumb_worker, [(str(cache / f"{k}.npy"), fr) for k in m["key"]], chunksize=64)))
    np.save(out_dir / "thumbs.npy", thumbs)

    ref = (m["axcodes"] == "LAS") & m["affine_consistent"].astype(bool)
    ref_idx = np.flatnonzero(ref.values)
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(ref_idx)
    n_val = len(ref_idx) // 10
    tr_idx, va_idx = ref_idx[n_val:], ref_idx[:n_val]
    dev = "cuda"
    net = orientation_net().to(dev)
    X = torch.from_numpy(thumbs).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)

    def batch(idx, augment):
        x = X[idx]
        k = torch.randint(0, 4, (len(idx),), device=dev)
        if augment:
            flip = torch.rand(len(idx), device=dev) < 0.5           # mirror-invariant by construction
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            x = x + 0.03 * torch.randn_like(x)
        x = torch.stack([torch.rot90(xi, int(ki), dims=(-2, -1)) for xi, ki in zip(x, k)])
        return x, k

    for ep in range(15):
        net.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), 256):
            x, k = batch(perm[i:i + 256], True)
            loss = F.cross_entropy(net(x), k)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            x, k = batch(va_idx, False)
            acc = (net(x).argmax(1) == k).float().mean().item()
        print(f"orientation epoch {ep} loss {loss.item():.4f} val acc {acc:.4f}")

    net.eval()
    with torch.no_grad():
        probs = torch.cat([F.softmax(net(X[i:i + 512]), 1) for i in range(0, len(X), 512)]).cpu().numpy()
    m["orient_pred"] = probs.argmax(1)
    m["orient_conf"] = probs.max(1)
    # The predicted class k means "the scan currently looks like canonical rotated by k";
    # undo by rotating with -k.
    m["orient_fix_k"] = (-m["orient_pred"]) % 4
    m["orient_uncertain"] = m["orient_conf"] < 0.9
    m["is_ref"] = ref.values
    m["orient_status"] = np.where(m["orient_uncertain"], "uncertain",
                                  np.where(m["orient_fix_k"] == 0, "ok", "fixed"))
    cols = ["key", "nifti_path", "axcodes", "affine_consistent", "is_ref", "orient_pred", "orient_conf",
            "orient_fix_k", "orient_status"]
    m[cols].to_csv(out_dir / "orientation.csv", index=False)
    rev = m[m["orient_uncertain"] | ((m["orient_fix_k"] != 0) & m["is_ref"]) | ~m["axcodes"].isin(["LAS", "LSP"])]
    rev[cols].to_csv(out_dir / "orientation_review.csv", index=False)
    print(m.groupby(["axcodes", "orient_status"]).size().to_string())
    print(f"review list: {len(rev)} scans")
    torch.save(net.state_dict(), out_dir / "orientation_net.pt")

    fix = m[(m["orient_fix_k"] != 0) & ~m["orient_uncertain"]]
    for _, r in fix.iterrows():
        f = cache / f"{r['key']}.npy"
        arr = np.load(f)
        np.save(f, np.ascontiguousarray(np.rot90(arr, int(r["orient_fix_k"]), axes=(-2, -1))))
    print(f"rotated {len(fix)} cached scans in place")
    _review_montages(rev, cache, out_dir)


def _thumb_worker(args):
    path, fracs = args
    return orientation_thumb(np.load(path, mmap_mode="r"), fracs=fracs)


def _review_montages(rev, cache, out_dir, per_page=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rev = rev.reset_index(drop=True)
    for page in range(int(np.ceil(len(rev) / per_page))):
        sub = rev.iloc[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(int(np.ceil(len(sub) / 6)), 6, figsize=(15, 2.7 * np.ceil(len(sub) / 6)),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
            arr = np.load(cache / f"{r['key']}.npy", mmap_mode="r")
            ax.imshow(arr[len(arr) // 3, 0], cmap="gray")
            ax.set_title(f"{r['key']} {r['axcodes']} {r['orient_status']} c={r['orient_conf']:.2f}", fontsize=7)
        fig.suptitle("Orientation review (after fix; anterior should be UP)")
        fig.savefig(out_dir / f"review_{page:02d}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------- dataset plots

def stage_plots(cfg, n_check=6):
    from utils import viz
    seed_everything(cfg["seed"])
    classes = cfg["classes"]
    out = p(cfg, "plots") / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    cache = p(cfg, "slice_cache")
    from data.dataset import load_table
    m = pd.read_csv(p(cfg, "manifest"))
    m["key"] = m["accession"].astype(int).astype(str) + "_" + m["ct_index"].astype(int).astype(str)
    d = load_table(cfg, require_split=False)

    viz.plot_class_distribution({"pre-surgery cohort": d[classes].values}, classes, out / "class_distribution_all.png")
    viz.plot_label_cooccurrence(d[classes].values, classes, out / "label_cooccurrence.png")
    viz.plot_bag_statistics(d["n_slices"], d["slice_thickness"], d["mask_area_mean_cm2"],
                            out / "bag_statistics.png", kept_slices=d["n_kept"])
    hcols = [c for c in d.columns if c.startswith("hist_")]
    hp = {c: d.loc[d[c] == 1, hcols].mean().values for c in classes}
    hn = {c: d.loc[d[c] == 0, hcols].mean().values for c in classes}
    viz.plot_hu_histograms(hp, hn, cfg["preprocess"]["hu_hist_bins"], classes, out / "hu_histograms.png")

    rng = np.random.default_rng(cfg["seed"])
    for key in rng.choice(d["key"].values, 12, replace=False):
        arr = np.load(cache / f"{key}.npy")
        r = d[d["key"] == key].iloc[0]
        lab = ",".join(c for c in classes if r[c] == 1) or "none"
        viz.plot_scan_montage(arr, out / "montages" / f"{key}.png", title=f"{key} [{lab}] {r['axcodes']}")
    mm = m[m["usable"]].copy()
    for key in rng.choice(d["key"].values, n_check, replace=False):
        row = mm[mm["key"] == key].iloc[0].to_dict()
        s, dbg = process_scan(row, cfg, cache, debug=True)
        z = len(dbg["keep"]) // 2
        viz.plot_preprocessing_check(dbg["hu"][z], dbg["gen"][z], dbg["out"][z].astype(np.float32) / 255.0,
                                     out / "preprocessing_check" / f"{key}.png",
                                     title=f"{key} slice {dbg['keep'][z]} (pre-orientation-fix)")
    examples = {}
    for c in classes:
        pos = d[d[c] == 1].sample(6, random_state=cfg["seed"])
        examples[c] = []
        for _, r in pos.iterrows():
            arr = np.load(cache / f"{r['key']}.npy", mmap_mode="r")
            examples[c].append((np.array(arr[len(arr) // 2]), r["key"]))
    viz.plot_positive_examples_per_class(examples, classes, out / "positive_examples_per_class.png")
    flag_mask_outliers(d, p(cfg, "outputs") / "mask_area_outliers.csv")
    print("dataset plots written to", out)


def flag_mask_outliers(d, path):
    out = []
    for col in ["mask_area_mean_cm2", "intracranial_vol_cm3", "n_kept"]:
        q1, q3 = d[col].quantile([0.25, 0.75])
        lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
        bad = d[(d[col] < lo) | (d[col] > hi)]
        out.append(bad.assign(outlier_metric=col)[["key", "outlier_metric", col, "nifti_path"]])
    res = pd.concat(out)
    res.to_csv(path, index=False)
    print(f"mask-area outliers: {res['key'].nunique()} scans -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "trim", "orient", "plots", "stats"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    if args.stage == "cache":
        stage_cache(cfg, args.overwrite, args.limit)
    elif args.stage == "trim":
        stage_trim(cfg)
    elif args.stage == "orient":
        stage_orient(cfg)
    elif args.stage == "plots":
        stage_plots(cfg)
    else:
        collect_stats(cfg)


if __name__ == "__main__":
    main()
