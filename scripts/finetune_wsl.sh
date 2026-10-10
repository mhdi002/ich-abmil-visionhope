#!/usr/bin/env bash
# Continue fine-tuning e2e_vh (encoder + head) on the anatomically trimmed bags, then evaluate.
#   bash env_wsl.sh bash scripts/finetune_wsl.sh
# RTX 3070 8 GB: grad checkpointing on (60 slices = 2.4 GiB at the same speed), original batching kept
# (4 bags/step x 4 accum = 16 bags). Lower LRs than the first Stage B run (1e-5 / 1e-4): continued training.
set -euo pipefail
PY=~/envs/ich/bin/python
RUNS="/mnt/d/All parts NIFTI files/ich_pipeline/outputs/runs"
mkdir -p "$RUNS/e2e_vh_oldbags_newtrim"
# original weights scored on the new bags (baseline); no cached preds so eval recomputes them
cp e2e_vh/best.pt "$RUNS/e2e_vh_oldbags_newtrim/best.pt"
$PY -u eval.py --run e2e_vh_oldbags_newtrim

$PY -u train.py --stage B --run e2e_vh_ft --init e2e_vh/best.pt --set \
  train.stage_b.epochs=8 train.stage_b.lr_backbone=5.0e-6 train.stage_b.lr_head=3.0e-5 \
  train.stage_b.warmup_epochs=0.5 train.stage_b.patience=3 train.stage_b.grad_checkpointing=true \
  train.stage_b.slice_budget=60 train.stage_b.bags_per_step=4 train.stage_b.grad_accum=4 \
  train.num_workers=4

$PY -u eval.py --run e2e_vh_ft
$PY -u eval.py --run e2e_vh_ft --threshold-mode recall --target-recall 0.8
echo FINETUNE_DONE
