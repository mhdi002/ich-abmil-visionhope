#!/usr/bin/env bash
# De-identified galleries + eval plots of e2e_vh_ft into docs/figures (no scan keys).
#   bash env_wsl.sh bash scripts/export_ft_docs_wsl.sh
set -euo pipefail
PY=~/envs/ich/bin/python
OUT="/mnt/d/All parts NIFTI files/ich_pipeline/outputs"
$PY -u scripts/export_public_galleries.py e2e_vh_ft docs/figures/eval/e2e_vh_ft
for run in e2e_vh_ft e2e_vh_ft_recall80; do
  mkdir -p docs/figures/eval/$run
  cp "$OUT/plots/eval/$run"/{train,val,test}_*.png docs/figures/eval/$run/
done
mkdir -p docs/figures/train
cp "$OUT/plots/train/e2e_vh_ft/training_curves.png" docs/figures/train/e2e_vh_ft_training_curves.png 2>/dev/null || ls "$OUT/plots/train/e2e_vh_ft/"
echo EXPORT_DONE
