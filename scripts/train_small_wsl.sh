#!/usr/bin/env bash
# One VisionHOPE-small model for everything: 5 hemorrhage classes + midline shift (mm and > 0 / > 5 mm) + cisterns +
# Marshall components + mass effect / edema / laterality, trained end-to-end from the ImageNet weights.
# Dual bag: the brain slices (anatomic trim) and the basal-cistern slices of the same scan go through the encoder
# together; the cistern heads attend only to the cistern slices, every other head only to the brain slices.
#   bash env_wsl.sh bash scripts/train_small_wsl.sh
# Epoch 1 trained the fresh head on a frozen encoder (replaces the Stage A feature cache); that checkpoint is in
# runs/vh_small_all_e1 and this script resumes from it with the encoder trainable (9 more epochs). Encoder LR 1e-5
# as in the first ImageNet -> CT run (1e-4 collapsed the features). Midline-shift mm loss 5x on shifted scans.
# Self-modifying learning: the encoder's SRNL memories train at 5x the encoder LR (they barely moved at 1x),
# SRNL also runs over the slice sequence in the head (model.srnl_context), and the class / task / regression loss
# weights are learned (tasks.uncertainty_weighting) so no task group stalls the others.
# Memory: 160-slice passes crashed once the encoder trained (WSL reports the OOM as "CUDA device not ready");
# 96-slice passes with checkpointing, 8 bags x 2 accumulation = 16 scans per step. ~2 h/epoch on an RTX 3070.
set -euo pipefail
PY=~/envs/ich/bin/python
INIT="/mnt/d/All parts NIFTI files/ich_pipeline/outputs/runs/vh_small_all_e1/best.pt"

$PY -u train.py --stage B --run vh_small_all --init "$INIT" --set tasks.enabled=true model.dual_bag=true \
  encoder.name=visionhope_small encoder.checkpoint=weights/visionhope_small.pth \
  train.stage_b.epochs=9 train.stage_b.freeze_encoder_epochs=0 train.stage_b.ema_warmup=true \
  train.stage_b.lr_backbone=1.0e-5 train.stage_b.lr_head=2.0e-4 train.stage_b.warmup_epochs=0.5 \
  train.stage_b.patience=3 train.stage_b.grad_checkpointing=true tasks.regression_pos_weight=5.0 \
  train.stage_b.lr_srnl_mult=5.0 model.srnl_context=true tasks.uncertainty_weighting=true \
  train.stage_b.slice_budget=96 train.stage_b.bags_per_step=8 train.stage_b.grad_accum=2 \
  train.num_workers=4

$PY -u eval.py --run vh_small_all
$PY -u scripts/eval_marshall.py --run vh_small_all
$PY -u scripts/calibrate.py --run vh_small_all     # -> calibration.json for predict_scan.py
echo SMALL_DONE
