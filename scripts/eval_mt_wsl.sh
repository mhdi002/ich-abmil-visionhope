#!/usr/bin/env bash
# Evaluate the multi-task run: 5-class report + task heads / midline shift / Marshall rule.
#   bash env_wsl.sh bash scripts/eval_mt_wsl.sh
set -euo pipefail
PY=~/envs/ich/bin/python
$PY -u eval.py --run e2e_vh_mt
$PY -u scripts/eval_marshall.py --run e2e_vh_mt
echo MT_EVAL_DONE
