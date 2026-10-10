#!/usr/bin/env bash
# Run a project script under env_wsl.sh with its output in <outputs>/logs/<name>.log
#   wsl -d Ubuntu-24.04 -- bash scripts/wsl_run.sh <name> <command...>
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="/mnt/d/All parts NIFTI files/ich_pipeline/outputs/logs"
mkdir -p "$LOGS"
name="$1"; shift
bash "$ROOT/env_wsl.sh" "$@" > "$LOGS/$name.log" 2>&1
code=$?
echo "EXIT=$code"
tail -40 "$LOGS/$name.log"
exit $code
