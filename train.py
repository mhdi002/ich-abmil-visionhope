"""Training: frozen-feature cache, Stage A (MIL head on cached features), Stage B (end-to-end).

  python train.py --stage features                       # cache encoder features (clean + 2 aug views)
  python train.py --stage A --run abmil_vh               # train ABMIL head on cached features
  python train.py --stage B --run e2e_vh --init outputs/runs/abmil_vh/best.pt
Overrides: --set model.pooling=mean loss.name=bce ...
"""
import argparse
import copy
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import (FeatureBagDataset, SliceBagDataset, balanced_sampler, gpu_prepare, load_slices,
                          load_table, pad_collate, task_names)
from models.abmil import BagModel, build_head, load_head_partial
from models.encoder import build_encoder, encode_bag
from models.losses import build_loss
from utils import viz
from utils.config import load_config, p, parse_overrides, seed_everything

HISTORY_KEYS = ["train_loss", "val_loss", "train_auc", "val_auc", "train_auprc", "val_auprc",
                "lr", "attn_entropy", "attn_peak", "grad_norm",
                "train_acc", "val_acc", "val_bal_acc", "train_recall", "val_recall", "val_precision", "val_specificity"]


# ---------------------------------------------------------------- helpers

class EMA:
    def __init__(self, model, decay, warmup=False):
        self.decay, self.warmup, self.steps = decay, warmup, 0
        self.model = copy.deepcopy(model).eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps)) if self.warmup else self.decay
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def cosine_warmup(opt, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def attention_stats(attn_list):
    """attn_list: per scan [C,n]. Returns normalized entropy and peak, per class and mean."""
    ent, peak = [], []
    for a in attn_list:
        n = a.shape[1]
        if n < 2:
            continue
        h = -(a * np.log(np.clip(a, 1e-12, 1))).sum(1) / np.log(n)
        ent.append(h)
        peak.append(a.max(1))
    ent, peak = np.array(ent), np.array(peak)
    return ent.mean(0), peak.mean(0)


def per_class_metrics(y, prob, classes):
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {}
    for c, name in enumerate(classes):
        if 0 < y[:, c].sum() < len(y):
            out[name] = (float(roc_auc_score(y[:, c], prob[:, c])), float(average_precision_score(y[:, c], prob[:, c])))
        else:
            out[name] = (float("nan"), float("nan"))
    return out


def loss_fn_total(criterion, out, y, aux, aux_w, t=None, r=None, task_cfg=None):
    l_cls = criterion(out["logits"], y)
    if aux_w and "aux_logit" in out:
        l_cls = l_cls + aux_w * F.binary_cross_entropy_with_logits(out["aux_logit"].float(), aux)
    l_bin = l_reg = None
    if task_cfg and t is not None and out.get("task_logits") is not None and t.shape[1]:
        l_bin = F.binary_cross_entropy_with_logits(out["task_logits"].float(), t)
    if task_cfg and r is not None and out.get("reg") is not None and r.shape[1]:
        # most scans have 0 mm shift: up-weight the shifted ones so large shifts are not regressed toward 0
        w = 1.0 + (float(task_cfg.get("regression_pos_weight", 1.0)) - 1.0) * (r > 0).float()
        l_reg = (w * F.smooth_l1_loss(out["reg"].float(), r, reduction="none")).sum() / w.sum()
    if "log_vars" in out:
        # learned balance: each group weighted by exp(-s), +s regularizer; s clamped for stability
        s = out["log_vars"].float().clamp(-4, 4)
        loss = torch.exp(-s[0]) * l_cls + s[0]
        for k, li in ((1, l_bin), (2, l_reg)):
            if li is not None:
                loss = loss + torch.exp(-s[k]) * li + s[k]
        return loss
    loss = l_cls
    if l_bin is not None:
        loss = loss + task_cfg["weight_binary"] * l_bin
    if l_reg is not None:
        loss = loss + task_cfg["weight_regression"] * l_reg
    return loss


def task_cfg_of(cfg):
    t = cfg.get("tasks") or {}
    return t if t.get("enabled") else None


def check_attention(ent, peak, classes, log):
    for c, name in enumerate(classes):
        if ent[c] > 0.97:
            log(f"  WARNING attention for {name} is near-uniform (entropy {ent[c]:.3f})")
        if peak[c] > 0.9:
            log(f"  WARNING attention for {name} collapsed to one slice (peak {peak[c]:.3f})")


def make_logger(run_dir):
    f = open(run_dir / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()
    return log


def pick_snapshot_scans(df, classes, seed, n=4):
    rng = np.random.default_rng(seed)
    picks = []
    for c in classes[:3]:
        cand = df[(df[c] == 1)]
        picks.append(cand.iloc[rng.integers(len(cand))]["key"])
    neg = df[df[classes].sum(1) == 0]
    picks.append(neg.iloc[rng.integers(len(neg))]["key"])
    return picks[:n]


def snapshot(cfg, keys, res, classes, out_path, writer, epoch):
    cache = p(cfg, "slice_cache")
    idx = {k: i for i, k in enumerate(res["keys"])}
    scans = []
    for k in keys:
        if k not in idx:
            continue
        i = idx[k]
        scans.append(dict(slices=load_slices(cache, k), attn=res["attn"][i],
                          labels=res["y"][i], probs=res["prob"][i], name=k))
    if scans:
        viz.plot_attention_snapshot(scans, classes, out_path, writer, epoch)


# ---------------------------------------------------------------- feature cache

def feature_dir(cfg):
    return p(cfg, "feature_cache") / cfg["encoder"]["name"]


def extract_features(cfg, views=("clean", "aug1", "aug2"), keys=None):
    seed_everything(cfg["seed"])
    df = load_table(cfg)
    if keys is not None:
        df = df[df["key"].isin(keys)]
    enc = build_encoder(cfg).cuda().eval()
    root = feature_dir(cfg)
    for view in views:
        sub = df if view == "clean" else df[df["split"] == "train"]
        out = root / view
        out.mkdir(parents=True, exist_ok=True)
        todo = sub[[not (out / f"{k}.npy").exists() for k in sub["key"]]]
        print(f"[features] {cfg['encoder']['name']} view={view}: {len(todo)} / {len(sub)} scans to encode", flush=True)
        if len(todo) == 0:
            continue
        dl = DataLoader(SliceBagDataset(todo, cfg), batch_size=1, num_workers=8, collate_fn=pad_collate,
                        pin_memory=True, prefetch_factor=4)
        t0 = time.time()
        for i, b in enumerate(dl):
            key = b["key"][0]
            x = b["x"][0].cuda(non_blocking=True)
            g = torch.Generator().manual_seed(zlib.crc32(f"{key}_{view}".encode()))
            x = gpu_prepare(x, cfg, train=(view != "clean"), generator=g)
            h = encode_bag(enc, x).cpu().numpy().astype(np.float16)
            np.save(out / f"{key}.npy", h)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- evaluation loops

@torch.no_grad()
def eval_features(head, loader, criterion, cfg):
    head.eval()
    ys, probs, auxp, keys, attn, losses = [], [], [], [], [], []
    for b in loader:
        x, mask = b["x"].cuda(non_blocking=True), b["mask"].cuda()
        y, aux = b["y"].cuda(), b["aux"].cuda()
        out = head(x, mask)
        losses.append(loss_fn_total(criterion, out, y, aux, cfg["loss"]["aux_weight"]).item() * len(y))
        ys.append(y.cpu().numpy())
        probs.append(torch.sigmoid(out["logits"].float()).cpu().numpy())
        if "aux_logit" in out:
            auxp.append(torch.sigmoid(out["aux_logit"].float()).cpu().numpy())
        keys += b["key"]
        a = out["attn"].float().cpu().numpy()
        for i, n in enumerate(b["mask"].sum(1).tolist()):
            attn.append(a[i, :, :n])
    y = np.concatenate(ys)
    return dict(y=y, prob=np.concatenate(probs), aux_prob=np.concatenate(auxp) if auxp else None,
                keys=keys, attn=attn, loss=float(np.sum(losses) / len(y)))


@torch.no_grad()
def eval_bags(model, loader, criterion, cfg, chunk=64):
    """End-to-end evaluation: encoder in eval mode on all slices of each bag."""
    model.eval()
    ys, probs, auxp, keys, attn, losses = [], [], [], [], [], []
    tys, tps, rys, rps, tattn = [], [], [], [], []
    tc = task_cfg_of(cfg)
    for b in loader:
        x = b["x"][0].cuda(non_blocking=True)
        y, aux = b["y"].cuda(), b["aux"].cuda()
        t, r = b.get("t"), b.get("r")
        t, r = (t.cuda() if t is not None else None), (r.cuda() if r is not None else None)
        xin = gpu_prepare(x, cfg, train=False)
        h = encode_bag(model.encoder, xin, chunk)[None]
        mask = torch.ones(1, h.shape[1], dtype=torch.bool, device=h.device)
        grp = b["grp"][:, :h.shape[1]].cuda() if "grp" in b else None
        out = model.head(h, mask) if grp is None else model.head(h, mask, grp)
        losses.append(loss_fn_total(criterion, out, y, aux, cfg["loss"]["aux_weight"], t, r, tc).item())
        ys.append(y.cpu().numpy())
        probs.append(torch.sigmoid(out["logits"].float()).cpu().numpy())
        if "aux_logit" in out:
            auxp.append(torch.sigmoid(out["aux_logit"].float()).cpu().numpy())
        keys += b["key"]
        attn.append(out["attn"][0].float().cpu().numpy())
        if tc and "task_logits" in out:
            tys.append(t.cpu().numpy())
            tps.append(torch.sigmoid(out["task_logits"].float()).cpu().numpy())
            rys.append(r.cpu().numpy())
            rps.append(out["reg"].float().cpu().numpy())
            tattn.append(out["task_attn"][0].float().cpu().numpy())
    y = np.concatenate(ys)
    res = dict(y=y, prob=np.concatenate(probs), aux_prob=np.concatenate(auxp) if auxp else None,
               keys=keys, attn=attn, loss=float(np.mean(losses)))
    if tys:
        res.update(task_y=np.concatenate(tys), task_prob=np.concatenate(tps), reg_y=np.concatenate(rys),
                   reg_pred=np.concatenate(rps), task_attn=tattn)
    return res


# ---------------------------------------------------------------- epoch bookkeeping

class Tracker:
    def __init__(self, cfg, run_dir, writer, log):
        self.cfg, self.run_dir, self.writer, self.log = cfg, run_dir, writer, log
        self.classes = cfg["classes"]
        self.history = {k: [] for k in HISTORY_KEYS}
        self.per_class_auc = {c: [] for c in self.classes}
        self.per_class_recall = {c: [] for c in self.classes}
        self.best, self.best_epoch, self.bad = -1.0, -1, 0

    def update(self, epoch, train_loss, grad_norm, lr, tr, va):
        tr_auc, tr_ap = viz.macro_metrics(tr["y"], tr["prob"])
        va_auc, va_ap = viz.macro_metrics(va["y"], va["prob"])
        ent, peak = attention_stats(va["attn"])
        _, trm = viz.threshold_metrics(tr["y"], tr["prob"], 0.5)
        va_per, vam = viz.threshold_metrics(va["y"], va["prob"], 0.5)
        for k, v in zip(HISTORY_KEYS, [train_loss, va["loss"], tr_auc, va_auc, tr_ap, va_ap, lr,
                                       float(ent.mean()), float(peak.mean()), grad_norm,
                                       trm["accuracy"], vam["accuracy"], vam["balanced_accuracy"], trm["recall"],
                                       vam["recall"], vam["precision"], vam["specificity"]]):
            self.history[k].append(float(v))
        pcm = per_class_metrics(va["y"], va["prob"], self.classes)
        for i, c in enumerate(self.classes):
            self.per_class_auc[c].append(pcm[c][0])
            self.per_class_recall[c].append(float(va_per[i]["recall"]))
        if self.writer is not None:
            for c, (a, ap) in pcm.items():
                self.writer.add_scalar(f"val_auprc/{c}", ap, epoch)
            for c, (e, pk) in enumerate(zip(ent, peak)):
                self.writer.add_scalar(f"attn_entropy/{self.classes[c]}", e, epoch)
                self.writer.add_scalar(f"attn_peak/{self.classes[c]}", pk, epoch)
        viz.save_epoch_plots(self.history, self.per_class_auc, self.classes, self.cfg["paths"]["plots_train"],
                             self.writer, epoch, per_class_recall=self.per_class_recall)
        self.log(f"epoch {epoch:3d} | loss {train_loss:.4f}/{va['loss']:.4f} | AUROC train {tr_auc:.4f} val {va_auc:.4f} "
                 f"| AUPRC train {tr_ap:.4f} val {va_ap:.4f} | ent {ent.mean():.3f} peak {peak.mean():.3f} "
                 f"| gn {grad_norm:.2f} lr {lr:.2e}")
        self.log(f"   @0.5 acc train {trm['accuracy']:.3f} val {vam['accuracy']:.3f} (bal {vam['balanced_accuracy']:.3f}) "
                 f"| recall train {trm['recall']:.3f} val {vam['recall']:.3f} | val prec {vam['precision']:.3f} "
                 f"spec {vam['specificity']:.3f}")
        self.log("   val AUROC/AUPRC " + " ".join(f"{c}={a:.3f}/{ap:.3f}" for c, (a, ap) in pcm.items()))
        self.log("   val acc/recall@0.5 " + " ".join(f"{c}={d['accuracy']:.3f}/{d['recall']:.3f}"
                                                  for c, d in zip(self.classes, va_per)))
        check_attention(ent, peak, self.classes, self.log)
        select = va_ap
        if "task_y" in va:
            tb, tr_names = task_names(self.cfg)
            tm = per_class_metrics(va["task_y"], va["task_prob"], tb)
            sel = self.cfg["train"].get("select_tasks") or tb     # e.g. only the cistern heads
            task_ap = float(np.nanmean([tm[k][1] for k in sel]))
            self.log("   val tasks AUROC/AUPRC " + " ".join(f"{k}={a:.3f}/{ap:.3f}" for k, (a, ap) in tm.items()))
            for j, name in enumerate(tr_names):
                scale = float(self.cfg["tasks"]["regression"][name][1])
                err = np.abs(va["reg_pred"][:, j] - va["reg_y"][:, j]) * scale
                pos = va["reg_y"][:, j] > 0
                cc = np.corrcoef(va["reg_pred"][:, j], va["reg_y"][:, j])[0, 1]
                self.log(f"   val {name}: MAE {err.mean():.2f}, MAE on positives {err[pos].mean():.2f}, r {cc:.3f}")
            self.history.setdefault("val_task_auprc", []).append(task_ap)
            select = 0.5 * (va_ap + task_ap)
            if self.cfg["train"].get("select_tasks"):
                select = task_ap
                self.log(f"   selection: task AUPRC over {sel} = {select:.4f}")
            else:
                self.log(f"   selection: 0.5 x (class AUPRC {va_ap:.4f} + task AUPRC {task_ap:.4f}) = {select:.4f}")
        (self.run_dir / "history.json").write_text(json.dumps(
            dict(history=self.history, per_class_auc=self.per_class_auc, per_class_recall=self.per_class_recall),
            indent=1))
        improved = select > self.best
        if improved:
            self.best, self.best_epoch, self.bad = select, epoch, 0
        else:
            self.bad += 1
        return improved


def setup_run(cfg, run):
    run_dir = p(cfg, "outputs") / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["plots_train"] = str(p(cfg, "plots") / "train" / run)
    Path(cfg["paths"]["plots_train"]).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    writer = SummaryWriter(str(p(cfg, "tensorboard") / run))
    return run_dir, writer, make_logger(run_dir)


# ---------------------------------------------------------------- Stage A

def train_stage_a(cfg, run, train_df=None, val_df=None, eval_train_df=None, quiet_plots=False):
    seed_everything(cfg["seed"])
    classes = cfg["classes"]
    df = load_table(cfg)
    train_df = df[df["split"] == "train"] if train_df is None else train_df
    val_df = df[df["split"] == "val"] if val_df is None else val_df
    eval_train_df = train_df if eval_train_df is None else eval_train_df
    run_dir, writer, log = setup_run(cfg, run)
    sa = cfg["train"]["stage_a"]
    fdir = feature_dir(cfg)
    views = [v for v in ("clean", "aug1", "aug2") if (fdir / v).exists()]
    log(f"[stage A] run={run} encoder={cfg['encoder']['name']} pooling={cfg['model']['pooling']} "
        f"attention={cfg['model']['attention']} srnl={cfg['model']['srnl_context']} loss={cfg['loss']['name']} "
        f"views={views} train={len(train_df)} val={len(val_df)}")

    ds_tr = FeatureBagDataset(train_df, cfg, fdir, views, train=True)
    sampler = balanced_sampler(ds_tr.y, seed=cfg["seed"]) if cfg["train"]["balanced_sampler"] else None
    dl_tr = DataLoader(ds_tr, batch_size=sa["batch_size"], sampler=sampler, shuffle=sampler is None,
                       num_workers=cfg["train"]["num_workers"], collate_fn=pad_collate, drop_last=len(ds_tr) > sa["batch_size"],
                       persistent_workers=True)
    mk = lambda d: DataLoader(FeatureBagDataset(d, cfg, fdir, ["clean"]), batch_size=64, shuffle=False,  # noqa: E731
                              num_workers=cfg["train"]["num_workers"], collate_fn=pad_collate, persistent_workers=True)
    dl_tre, dl_va = mk(eval_train_df), mk(val_df)
    in_dim = np.load(fdir / "clean" / f"{train_df['key'].iloc[0]}.npy").shape[1]

    head = build_head(cfg, in_dim).cuda()
    criterion = build_loss(cfg, ds_tr.y).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=sa["lr_head"], weight_decay=sa["weight_decay"])
    steps = sa["epochs"] * max(len(dl_tr), 1)
    sched = cosine_warmup(opt, steps, sa["warmup_epochs"] * len(dl_tr))
    ema = EMA(head, cfg["train"]["ema_decay"])
    tracker = Tracker(cfg, run_dir, None if quiet_plots else writer, log)
    snap_keys = pick_snapshot_scans(val_df, classes, cfg["seed"])

    for epoch in range(1, sa["epochs"] + 1):
        head.train()
        tot, n, gns = 0.0, 0, []
        for b in dl_tr:
            x, mask = b["x"].cuda(non_blocking=True), b["mask"].cuda()
            y, aux = b["y"].cuda(), b["aux"].cuda()
            out = head(x, mask)
            loss = loss_fn_total(criterion, out, y, aux, cfg["loss"]["aux_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gns.append(float(torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)))
            opt.step()
            sched.step()
            ema.update(head)
            tot += loss.item() * len(y)
            n += len(y)
        tr = eval_features(ema.model, dl_tre, criterion, cfg)
        va = eval_features(ema.model, dl_va, criterion, cfg)
        improved = tracker.update(epoch, tot / n, float(np.mean(gns)), sched.get_last_lr()[0], tr, va)
        if improved:
            torch.save(dict(head=ema.model.state_dict(), cfg=cfg, in_dim=in_dim, epoch=epoch, stage="A"),
                       run_dir / "best.pt")
        if not quiet_plots and (epoch % cfg["train"]["attn_snapshot_every"] == 0 or epoch == 1):
            snapshot(cfg, snap_keys, va, classes, Path(cfg["paths"]["plots_train"]) / f"attention_epoch{epoch:03d}.png",
                     writer, epoch)
        if tracker.bad >= sa["patience"]:
            log(f"early stopping at epoch {epoch}; best val macro AUPRC {tracker.best:.4f} at epoch {tracker.best_epoch}")
            break
    log(f"[stage A] done. best epoch {tracker.best_epoch} val macro AUPRC {tracker.best:.4f}")
    writer.close()
    return tracker, run_dir


# ---------------------------------------------------------------- Stage B

def train_stage_b(cfg, run, init=None, train_df=None, val_df=None, eval_train_df=None):
    seed_everything(cfg["seed"])
    classes = cfg["classes"]
    df = load_table(cfg)
    train_df = df[df["split"] == "train"] if train_df is None else train_df
    val_df = df[df["split"] == "val"] if val_df is None else val_df
    sb = cfg["train"]["stage_b"]
    if sb.get("max_val_scans") and len(val_df) > sb["max_val_scans"]:
        val_df = val_df.sample(sb["max_val_scans"], random_state=cfg["seed"])
    if eval_train_df is None:
        eval_train_df = train_df.sample(min(len(train_df), 1000), random_state=cfg["seed"])
    run_dir, writer, log = setup_run(cfg, run)

    if "grad_checkpointing" in sb:
        cfg["encoder"]["grad_checkpointing"] = sb["grad_checkpointing"]
    encoder = build_encoder(cfg)
    head = build_head(cfg, encoder.out_dim)
    ck = torch.load(init, map_location="cpu", weights_only=False) if init else None
    if ck is not None and "head" in ck:
        head.load_state_dict(ck["head"])
        log(f"[stage B] head initialized from {init} (epoch {ck['epoch']})")
    model = BagModel(encoder, head)
    if ck is not None and "model" in ck:
        # continue fine-tuning an earlier Stage B model: encoder + head (head may gain task slots)
        load_head_partial(model, ck["model"], log)
        log(f"[stage B] encoder + head initialized from {init} (stage {ck.get('stage')}, epoch {ck['epoch']})")
    tb, tr = task_names(cfg)
    if tb or tr:
        log(f"[stage B] multi-task: {len(tb)} binary {tb}, regression {tr}")
    model = model.cuda()
    log(f"[stage B] run={run} encoder={cfg['encoder']['name']} train={len(train_df)} val={len(val_df)} "
        f"train-eval subset={len(eval_train_df)}")

    ds_tr = SliceBagDataset(train_df, cfg)
    sampler = balanced_sampler(ds_tr.y, seed=cfg["seed"]) if cfg["train"]["balanced_sampler"] else None
    nw = cfg["train"]["num_workers"]
    bps, budget = sb.get("bags_per_step", 1), sb.get("slice_budget", 10**9)
    dl_tr = DataLoader(ds_tr, batch_size=bps, sampler=sampler, shuffle=sampler is None, num_workers=nw,
                       collate_fn=pad_collate, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    mk = lambda d: DataLoader(SliceBagDataset(d, cfg), batch_size=1, shuffle=False, num_workers=nw,  # noqa: E731
                              collate_fn=pad_collate, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    dl_tre, dl_va = mk(eval_train_df), mk(val_df)

    enc_named = [(nm, q) for nm, q in model.encoder.named_parameters() if q.requires_grad]
    enc_params = [q for _, q in enc_named]
    # VisionHOPE's self-referential memories (SRNL) get their own, larger LR: at the plain encoder LR they barely
    # moved in earlier runs (0.03 % vs 0.1 % for the rest of the encoder)
    srnl_mult = float(sb.get("lr_srnl_mult", 1.0))
    srnl_p = [q for nm, q in enc_named if ".srnl." in nm]
    rest_p = [q for nm, q in enc_named if ".srnl." not in nm]
    log(f"[stage B] encoder params: {sum(q.numel() for q in rest_p) / 1e6:.1f} M at lr {sb['lr_backbone']:.1e}, "
        f"SRNL {sum(q.numel() for q in srnl_p) / 1e6:.2f} M at lr {sb['lr_backbone'] * srnl_mult:.1e}")
    opt = torch.optim.AdamW([dict(params=rest_p, lr=sb["lr_backbone"]),
                             dict(params=srnl_p, lr=sb["lr_backbone"] * srnl_mult),
                             dict(params=model.head.parameters(), lr=sb["lr_head"])], weight_decay=sb["weight_decay"])
    accum = sb["grad_accum"]
    steps = sb["epochs"] * math.ceil(len(dl_tr) / accum)
    sched = cosine_warmup(opt, steps, int(sb["warmup_epochs"] * len(dl_tr) / accum))
    criterion = build_loss(cfg, ds_tr.y).cuda()
    tc = task_cfg_of(cfg)
    ema = EMA(model, cfg["train"]["ema_decay"], warmup=bool(sb.get("ema_warmup")))
    tracker = Tracker(cfg, run_dir, writer, log)
    snap_keys = pick_snapshot_scans(val_df, classes, cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    frozen_epochs = int(sb.get("freeze_encoder_epochs", 0))

    for epoch in range(1, sb["epochs"] + 1):
        # head warm-up on a frozen encoder (fresh heads on a non-CT encoder; replaces the Stage A feature cache)
        frozen = epoch <= frozen_epochs
        for q in enc_params:
            q.requires_grad_(not frozen)
        if frozen:
            log(f"  epoch {epoch}: encoder frozen (head warm-up)")
        model.train()
        tot, n, gns, t0 = 0.0, 0, [], time.time()
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(dl_tr, 1):
            x, y, aux = b["x"].cuda(non_blocking=True), b["y"].cuda(), b["aux"].cuda()
            t, r = b["t"].cuda(), b["r"].cuda()
            grp_all = b["grp"].cuda() if "grp" in b else None
            lens = b["mask"].sum(1).tolist()
            B = len(lens)
            groups, cur, size = [], [], 0
            for j, L in enumerate(lens):
                if cur and size + L > budget:
                    groups.append(cur)
                    cur, size = [], 0
                cur.append(j)
                size += L
            groups.append(cur)
            for g in groups:
                n_max = max(lens[j] for j in g)
                xin = x.new_zeros((len(g), n_max, 3, cfg["encoder"]["input_size"], cfg["encoder"]["input_size"]),
                                  dtype=torch.float32)
                mask = torch.zeros(len(g), n_max, dtype=torch.bool, device=x.device)
                for k, j in enumerate(g):
                    xin[k, :lens[j]] = gpu_prepare(x[j, :lens[j]], cfg, train=True, generator=gen)
                    mask[k, :lens[j]] = True
                grp = grp_all[g][:, :n_max] if grp_all is not None else None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
                    out = model(xin, mask, grp)
                loss = loss_fn_total(criterion, out, y[g], aux[g], cfg["loss"]["aux_weight"], t[g], r[g], tc)
                (loss * len(g) / (B * accum)).backward()
                tot += loss.item() * len(g)
                n += len(g)
            if i % accum == 0 or i == len(dl_tr):
                gns.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
            if i % max(1, 500 // bps) == 0:
                log(f"  epoch {epoch} step {i}/{len(dl_tr)} loss {tot / n:.4f} {time.time() - t0:.0f}s "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        tr = eval_bags(ema.model, dl_tre, criterion, cfg)
        va = eval_bags(ema.model, dl_va, criterion, cfg)
        improved = tracker.update(epoch, tot / max(n, 1), float(np.mean(gns)), sched.get_last_lr()[-1], tr, va)
        if getattr(ema.model.head, "log_vars", None) is not None:
            wts = torch.exp(-ema.model.head.log_vars.detach().float().clamp(-4, 4)).tolist()
            log(f"   learned loss weights: classes {wts[0]:.3f}, binary tasks {wts[1]:.3f}, regression {wts[2]:.3f}")
        if improved:
            torch.save(dict(model=ema.model.state_dict(), cfg=cfg, epoch=epoch, stage="B"), run_dir / "best.pt")
        if epoch % max(1, cfg["train"]["attn_snapshot_every"] // 2) == 0 or epoch == 1:
            snapshot(cfg, snap_keys, va, classes, Path(cfg["paths"]["plots_train"]) / f"attention_epoch{epoch:03d}.png",
                     writer, epoch)
        log(f"  epoch time {time.time() - t0:.0f}s")
        if tracker.bad >= sb["patience"]:
            log(f"early stopping at epoch {epoch}; best val macro AUPRC {tracker.best:.4f} at epoch {tracker.best_epoch}")
            break
    log(f"[stage B] done. best epoch {tracker.best_epoch} val macro AUPRC {tracker.best:.4f}")
    writer.close()
    return tracker, run_dir


def dataset_plots_at_start(cfg):
    """Dataset plots required at the start of training (split distributions, augmentation preview)."""
    df = load_table(cfg)
    out = p(cfg, "plots") / "dataset"
    viz.plot_class_distribution({s: df.loc[df["split"] == s, cfg["classes"]].values for s in ("train", "val", "test")},
                                cfg["classes"], out / "class_distribution_splits.png")
    key = df[df["SDH"] == 1]["key"].iloc[0]
    x = torch.from_numpy(load_slices(p(cfg, "slice_cache"), key)).cuda()
    mid = len(x) // 2
    clean = gpu_prepare(x, cfg, train=False)[mid].cpu().numpy()
    g = torch.Generator().manual_seed(0)
    augs = [gpu_prepare(x, cfg, train=True, generator=g)[mid].cpu().numpy() for _ in range(5)]
    viz.plot_augmentation_preview(clean, augs, out / "augmentation_preview.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "A", "B"], required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--views", nargs="*", default=["clean", "aug1", "aug2"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    if args.stage == "features":
        extract_features(cfg, args.views)
    elif args.stage == "A":
        dataset_plots_at_start(cfg)
        train_stage_a(cfg, args.run or "stageA")
    else:
        dataset_plots_at_start(cfg)
        train_stage_b(cfg, args.run or "stageB", args.init)


if __name__ == "__main__":
    main()
