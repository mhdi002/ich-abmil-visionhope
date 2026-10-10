#!/usr/bin/env bash
# Multi-task Stage B from e2e_vh_ft: 5 hemorrhage classes + midline shift / cisterns / Marshall / mass effect /
# laterality heads on one VisionHOPE backbone, then evaluation.
#   bash env_wsl.sh bash scripts/finetune_mt_wsl.sh
# The 5 class slots start from e2e_vh_ft, the 18 new slots from scratch: head LR back to 1e-4, encoder 5e-6.
# Checkpoint selection: 0.5 x (class macro AUPRC + task macro AUPRC) on val.
set -euo pipefail
PY=~/envs/ich/bin/python

$PY -u train.py --stage B --run e2e_vh_mt --init e2e_vh_ft/best.pt --set tasks.enabled=true \
  train.stage_b.epochs=10 train.stage_b.lr_backbone=5.0e-6 train.stage_b.lr_head=1.0e-4 \
  train.stage_b.warmup_epochs=0.5 train.stage_b.patience=3 train.stage_b.grad_checkpointing=true \
  train.stage_b.slice_budget=60 train.stage_b.bags_per_step=4 train.stage_b.grad_accum=4 \
  train.num_workers=4

$PY -u eval.py --run e2e_vh_mt
$PY -u scripts/eval_marshall.py --run e2e_vh_mt
echo MT_DONE
