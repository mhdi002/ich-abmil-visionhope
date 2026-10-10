"""Score one CT scan: NIfTI in -> structured JSON report + Grad-CAM images out.

  python predict_scan.py path/to/scan.nii.gz --run vh_small_all --out reports/

Steps: header check -> preprocessing exactly as in training (anatomic trim for the brain slices, basal-cistern window
for the cistern slices) -> orientation CNN (fixes rotated scans) -> model (5 hemorrhage classes, midline shift in mm
and > 0 / > 5 mm, cisterns, Marshall components, mass effect / edema / laterality) -> HU lesion volume ->
Marshall CT score with the decision layer fitted on val (scripts/calibrate.py -> <run>/calibration.json) ->
Grad-CAM of every positive finding on its most-attended slices.

Research code: not a medical device and not clinically validated.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.dataset import gpu_prepare, task_names
from data.manifest import header_info
from lesion_volume import DEFAULTS as LESION_DEFAULTS, lesion_volumes
from models.abmil import BagModel, build_head
from models.encoder import build_encoder, encode_bag
from preprocess import orientation_net, orientation_thumb, process_scan
from scripts.calibrate import vi_prob_from
from utils.config import load_config, p

CAT_NAME = {1: "I", 2: "II", 3: "III", 4: "IV", 6: "VI"}
OTHER = ["mass_effect", "herniation", "edema", "hydrocephalus", "skull_fx"]
SIDES = [("SDH", "sdh"), ("EDH", "edh"), ("Contusion", "contusion")]


def load_model(run_dir):
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["encoder"]["grad_checkpointing"] = False
    enc = build_encoder(cfg)
    model = BagModel(enc, build_head(cfg, enc.out_dim))
    model.load_state_dict(ck["model"])
    return cfg, model.cuda().eval()


def preprocess(path, cfg):
    """Header -> two cached-format stacks (brain slices, cistern slices) as uint8 [N,3,256,256], plus the row."""
    hdr = header_info(str(path))
    if not hdr.get("header_ok"):
        raise ValueError(f"cannot read NIfTI header: {hdr.get('header_error')}")
    row = dict(hdr, accession=0, ct_index=0, nifti_path=str(path))
    pc = dict(cfg["preprocess"], trim_mode="anatomic")
    stats, dbg = process_scan(row, {**cfg, "preprocess": pc}, None, debug=True)
    if not stats.get("ok"):
        raise ValueError(f"preprocessing failed: {stats.get('reason')}")
    _, dbg_c = process_scan(row, {**cfg, "preprocess": {**pc, "trim_mode": "cistern"}}, None, debug=True)
    return row, dbg["out"], dbg_c["out"], stats


def fix_orientation(brain, cist, cfg, outputs):
    f = outputs / "orientation" / "orientation_net.pt"
    if not f.exists():
        return brain, cist, 0, None
    net = orientation_net()
    net.load_state_dict(torch.load(f, map_location="cpu"))
    net.eval()
    fr = tuple(cfg["preprocess"].get("orient_thumb_fracs", (0.25, 0.5, 0.75)))
    with torch.no_grad():
        pr = torch.softmax(net(torch.from_numpy(orientation_thumb(brain, fracs=fr))[None]), 1)[0].numpy()
    k = int((-pr.argmax()) % 4) if pr.max() >= 0.9 else 0
    rot = lambda a: np.ascontiguousarray(np.rot90(a, k, axes=(-2, -1)))  # noqa: E731
    return rot(brain), rot(cist), k, float(pr.max())


def slot_score(out, s, C, K):
    if s < C:
        return out["logits"][0, s]
    if s < C + K:
        return out["task_logits"][0, s - C]
    return out["reg"][0, s - C - K]


def gradcam(model, x, grp, slot, top_idx, C, K):
    """Grad-CAM of one head (any slot) on selected slices; x [N,3,H,W] prepared, grp [N]."""
    enc, acts = model.encoder, {}

    def hook(_, __, o):
        o.retain_grad()
        acts["a"] = o
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([enc(x[i:i + 32]).float() for i in range(0, len(x), 32)])
    handle = enc.last_stage().register_forward_hook(hook)
    try:
        with torch.enable_grad():
            h2 = h.clone()
            h2[top_idx] = enc(x[top_idx]).float()
            ones = torch.ones(1, len(h2), dtype=torch.bool, device=h2.device)
            out = model.head(h2[None], ones, grp[None])
            slot_score(out, slot, C, K).backward()
        a, g = acts["a"], acts["a"].grad
        cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1)).detach()
        cam = torch.nn.functional.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear")[:, 0]
        head_m = torch.nn.functional.max_pool2d((x[top_idx].amax(1, keepdim=True) > 0).float(), 7, 1, 3)[:, 0]
        cam = cam * head_m
        cam = cam / cam.flatten(1).max(1).values.clamp(min=1e-8)[:, None, None]
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    return cam.cpu().numpy()


def save_gradcam(x, cam, top_idx, attn, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(top_idx), figsize=(3.2 * len(top_idx), 3.6), squeeze=False)
    for ax, j, c in zip(axes[0], top_idx, cam):
        ax.imshow(x[j, 0], cmap="gray", vmin=0, vmax=1)
        ax.imshow(c, cmap="jet", alpha=0.35 * (c > 0.05), vmin=0, vmax=1)
        ax.set_title(f"slice {j}  attention {attn[j]:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--run", default="vh_small_final")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--no-gradcam", action="store_true")
    args = ap.parse_args()
    cur = load_config()
    outputs = p(cur, "outputs")
    run_dir = outputs / "runs" / args.run
    cal = json.loads((run_dir / "calibration.json").read_text())
    cfg, model = load_model(run_dir)
    classes = cfg["classes"]
    tb, tr = task_names(cfg)
    C, K = len(classes), len(tb)
    ti = {k: i for i, k in enumerate(tb)}
    scan = Path(args.scan)
    name = scan.name.replace(".nii.gz", "").replace(".nii", "")
    out_dir = Path(args.out) / name

    row, brain, cist, stats = preprocess(scan, cur)
    brain, cist, k, oconf = fix_orientation(brain, cist, cur, outputs)
    dual = bool(cfg["model"].get("dual_bag"))
    stack = np.concatenate([brain, cist]) if dual else brain
    grp = torch.as_tensor(np.r_[np.zeros(len(brain)), np.ones(len(cist) if dual else 0)], dtype=torch.long).cuda()
    x = gpu_prepare(torch.from_numpy(stack).cuda(), cfg, train=False)
    with torch.no_grad():
        h = encode_bag(model.encoder, x)[None]
        out = model.head(h, torch.ones(1, h.shape[1], dtype=torch.bool, device=h.device), grp[None])
    prob = torch.sigmoid(out["logits"].float())[0].cpu().numpy()
    tprob = torch.sigmoid(out["task_logits"].float())[0].cpu().numpy()
    reg = out["reg"].float()[0].cpu().numpy()
    attn = torch.cat([out["attn"], out["task_attn"]], 1)[0].float().cpu().numpy()      # [slots, N]

    vol = lesion_volumes(row, cur, {**LESION_DEFAULTS, **(cur.get("lesion_volume") or {})})
    vols = [vol.get(c, 0.0) for c in ("largest_cm3", "total_cm3", "extra_axial_cm3", "intra_axial_cm3")]
    th, tth, rt = cal["class_thresholds"], cal["task_thresholds"], cal["rule_thresholds"]
    pos = lambda name: bool(tprob[ti[name]] >= tth[name])  # noqa: E731
    mls_mm = float(max(0.0, reg[tr.index("mls_mm")] * cal["regression_scale"]["mls_mm"])) if "mls_mm" in tr else None

    comp = {"marshall_vi": float(vi_prob_from(cal, tprob[ti["marshall_vi"]], np.array(vols))[0]),
            "mls_gt5": float(tprob[ti["mls_gt5"]]), "cist_compressed": float(tprob[ti["cist_compressed"]]),
            "marshall_ge2": float(tprob[ti["marshall_ge2"]])}
    category = 1
    for head, cat in cal["rule_order"]:
        if comp[head] >= rt[head]:
            category = cat
            break
    cist_status = ("effaced" if pos("cist_effaced") else "compressed" if pos("cist_compressed") else "visible")
    report = {
        "scan": str(scan), "model": args.run, "encoder": cfg["encoder"]["name"],
        "preprocessing": {"brain_slices": int(len(brain)), "cistern_slices": int(len(cist)),
                          "slice_thickness_mm": round(stats["slice_mm"], 2),
                          "orientation_rotation_deg": 90 * k, "orientation_confidence": oconf},
        "hemorrhage": {c: {"probability": round(float(prob[i]), 4), "threshold": round(th[c], 4),
                           "present": bool(prob[i] >= th[c])} for i, c in enumerate(classes)},
        "midline_shift": {"mm": None if mls_mm is None else round(mls_mm, 1),
                          "probability_any": round(float(tprob[ti["mls_any"]]), 4),
                          "probability_gt_5mm": round(float(tprob[ti["mls_gt5"]]), 4),
                          "category": ">5 mm" if pos("mls_gt5") else ("0-5 mm" if pos("mls_any") else "none")},
        "cisterns": {"status": cist_status, "probability_compressed_or_worse": round(float(tprob[ti["cist_compressed"]]), 4),
                     "probability_effaced": round(float(tprob[ti["cist_effaced"]]), 4)},
        "lesion_volume": {"total_cm3": round(vol.get("total_cm3", 0.0), 1), "largest_cm3": round(vol.get("largest_cm3", 0.0), 1),
                          "extra_axial_cm3": round(vol.get("extra_axial_cm3", 0.0), 1),
                          "intra_axial_cm3": round(vol.get("intra_axial_cm3", 0.0), 1),
                          "patient_right_cm3": round(vol.get("patient_right_cm3", 0.0), 1),
                          "patient_left_cm3": round(vol.get("patient_left_cm3", 0.0), 1),
                          "n_lesions": vol.get("n_lesions", 0),
                          "method": "hyperdense voxels (HU), no segmentation labels: approximate, underestimates "
                                    "mixed-density lesions"},
        "laterality": {cls: {"right": round(float(tprob[ti[f"{s}_right"]]), 4), "left": round(float(tprob[ti[f"{s}_left"]]), 4)}
                       for cls, s in SIDES},
        "other_findings": {o: {"probability": round(float(tprob[ti[o]]), 4), "present": pos(o)} for o in OTHER},
        "marshall_ct": {"category": CAT_NAME[category], "components": {k2: round(v, 4) for k2, v in comp.items()},
                        "rule_thresholds": rt,
                        "note": "category V (surgically evacuated lesion) is not assessable from a pre-operative scan"},
        "gradcam": {},
        "disclaimer": "Research model; not a medical device and not clinically validated.",
    }

    if not args.no_gradcam:
        slot_names = list(classes) + tb + tr
        findings = [c for c in classes if report["hemorrhage"][c]["present"]]
        findings += [t for t in ("mls_gt5", "mls_any") if pos(t)][:1]
        findings += ["cist_compressed"] if pos("cist_compressed") else []
        for f in findings:
            s = slot_names.index(f)
            top = np.argsort(-attn[s])[:args.topk]
            top = top[attn[s][top] > 0]
            cam = gradcam(model, x, grp, s, torch.as_tensor(top, device="cuda"), C, K)
            fp = out_dir / f"gradcam_{f}.png"
            score = prob[s] if s < C else tprob[s - C]
            save_gradcam(x.float().cpu().numpy(), cam, top, attn[s], f"{f}  p={score:.2f}", fp)
            report["gradcam"][f] = str(fp)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
