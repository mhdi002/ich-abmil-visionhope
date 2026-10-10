#!/usr/bin/env bash
# Resume vh_small_all after the epoch-6 checkpoint (a transient short read from the D: drive killed epoch 7):
# 3 more epochs at the learning rates the original schedule had reached, decaying to 0. The epoch-6 checkpoint is
# kept in runs/vh_small_all_e6; whichever of the two has the better validation selection score is evaluated,
# calibrated and copied to runs/vh_small_final.
#   bash env_wsl.sh bash scripts/train_small_resume_wsl.sh
set -euo pipefail
PY=~/envs/ich/bin/python
RUNS="/mnt/d/All parts NIFTI files/ich_pipeline/outputs/runs"

$PY -u train.py --stage B --run vh_small_all_r --init "$RUNS/vh_small_all_e6/best.pt" --set tasks.enabled=true \
  model.dual_bag=true encoder.name=visionhope_small encoder.checkpoint=weights/visionhope_small.pth \
  train.stage_b.epochs=3 train.stage_b.freeze_encoder_epochs=0 train.stage_b.ema_warmup=false \
  train.stage_b.lr_backbone=3.0e-6 train.stage_b.lr_head=6.0e-5 train.stage_b.warmup_epochs=0.1 \
  train.stage_b.patience=3 train.stage_b.grad_checkpointing=true tasks.regression_pos_weight=5.0 \
  train.stage_b.lr_srnl_mult=5.0 model.srnl_context=true tasks.uncertainty_weighting=true \
  train.stage_b.slice_budget=96 train.stage_b.bags_per_step=8 train.stage_b.grad_accum=2 \
  train.num_workers=4

BEST=$($PY - <<'EOF'
import json
R = "/mnt/d/All parts NIFTI files/ich_pipeline/outputs/runs/"
def score(run):
    h = json.load(open(R + run + "/history.json"))["history"]
    return max(0.5 * (a + t) for a, t in zip(h["val_auprc"], h["val_task_auprc"]))
s6, sr = score("vh_small_all_e6"), score("vh_small_all_r")
print("vh_small_all_r" if sr > s6 else "vh_small_all_e6")
EOF
)
echo "best checkpoint: $BEST"
rm -rf "$RUNS/vh_small_final"; mkdir -p "$RUNS/vh_small_final"
cp "$RUNS/$BEST/best.pt" "$RUNS/$BEST/config.json" "$RUNS/$BEST/history.json" "$RUNS/vh_small_final/"
$PY -u eval.py --run vh_small_final
$PY -u scripts/eval_marshall.py --run vh_small_final
$PY -u scripts/calibrate.py --run vh_small_final     # -> calibration.json for predict_scan.py
echo SMALL_DONE
