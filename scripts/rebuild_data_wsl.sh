#!/usr/bin/env bash
# Rebuild the slice cache with the anatomic trim, fix orientation, and reuse the e2e_vh splits.
#   bash env_wsl.sh bash scripts/rebuild_data_wsl.sh
set -euo pipefail
PY=~/envs/ich/bin/python
$PY -u preprocess.py --stage cache
$PY -u preprocess.py --stage orient
$PY -u -m data.splits --from-preds e2e_vh
echo REBUILD_DONE
