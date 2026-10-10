#!/usr/bin/env bash
# Lesion volumes for val/test, then a cistern model on a basal-cistern slice window, then evaluation.
#   bash env_wsl.sh bash scripts/cistern_pipeline_wsl.sh
# The cistern model starts from e2e_vh_mt (same 23-slot head) and is trained on 15 mm below .. 25 mm above the
# skull-base trim line (preprocess.trim_mode=cistern, compressed cache in outputs_cistern/); its checkpoint is
# chosen on the cistern heads only. 8-slice bags: 8 bags/step x 2 accumulation = 16 bags.
set -euo pipefail
PY=~/envs/ich/bin/python
OC="/mnt/d/All parts NIFTI files/ich_pipeline/outputs_cistern"
CSET=(preprocess.trim_mode=cistern preprocess.cache_compress=true "paths.outputs=$OC"
      "paths.slice_cache=$OC/cache/slices" "paths.plots=$OC/plots" "paths.tensorboard=$OC/tb")

for split in val test; do
  [ -f "/mnt/d/All parts NIFTI files/ich_pipeline/outputs/lesion_volume/volumes_$split.csv" ] || \
    $PY -u lesion_volume.py --stage run --split $split --workers 6
  $PY -u lesion_volume.py --stage eval --split $split
done

$PY -u preprocess.py --stage cache --set "${CSET[@]}"
$PY -u scripts/apply_orientation.py --cache "$OC/cache/slices"

$PY -u train.py --stage B --run cistern_vh --init e2e_vh_mt/best.pt --set "${CSET[@]}" tasks.enabled=true \
  "train.select_tasks=[cist_compressed,cist_effaced]" \
  train.stage_b.epochs=8 train.stage_b.lr_backbone=5.0e-6 train.stage_b.lr_head=5.0e-5 \
  train.stage_b.warmup_epochs=0.5 train.stage_b.patience=3 train.stage_b.grad_checkpointing=true \
  train.stage_b.slice_budget=64 train.stage_b.bags_per_step=8 train.stage_b.grad_accum=2 \
  train.num_workers=4

$PY -u eval.py --run cistern_vh --set "${CSET[@]}"
$PY -u scripts/eval_marshall.py --run cistern_vh --set "${CSET[@]}"
echo CISTERN_PIPELINE_DONE
