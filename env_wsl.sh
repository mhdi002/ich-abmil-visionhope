# WSL2 (RTX 3070, sm_86) counterpart of env.sh: `source env_wsl.sh`, or `bash env_wsl.sh <command...>`
export ICH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_HOME="$HOME/envs/ich"                  # micromamba env: python 3.11 + conda-forge CUDA 12.4 nvcc
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions_ich"
export MAX_JOBS=8
export PYTHONPATH="$ICH_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ICH_ROOT"
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ $# -gt 0 ]; then exec "$@"; fi
