# Source before running anything: `source env.sh`
export ICH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_HOME="$HOME/cuda128"
export PATH="$CUDA_HOME/bin:$ICH_ROOT/.venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions_ich"
export MAX_JOBS=16
export PYTHONPATH="$ICH_ROOT:$PYTHONPATH"
