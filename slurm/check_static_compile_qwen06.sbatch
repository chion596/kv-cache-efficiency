#!/bin/bash

#SBATCH --job-name=kv-compile
#SBATCH --partition=test
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a

#SBATCH --time=00:20:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

# torch.compile / Triton могут создавать довольно большой cache.
# Не складываем его в home.
export TORCHINDUCTOR_CACHE_DIR="$KV_SCRATCH/torchinductor-cache"
export TRITON_CACHE_DIR="$KV_SCRATCH/triton-cache"

mkdir -p \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR"

echo "========================================"
echo "STATIC CACHE COMPILE SMOKE TEST"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "========================================"

python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))

print(
    "GPU capability:",
    torch.cuda.get_device_capability(0),
)
PY

python -u src/check_static_compile.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --context 2048 \
    --new-tokens 64 \
    --warmups 3 \
    --repeats 3

echo "========================================"
echo "FINISHED: $(date)"
echo "========================================"
