#!/bin/bash

#SBATCH --job-name=kv-comp-full
#SBATCH --partition=rocky
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a

#SBATCH --time=01:00:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

# Используем отдельный чистый compiler cache для каждого job.
# Так cold compilation не будет случайно брать артефакты
# предыдущего smoke-test.
export TORCHINDUCTOR_CACHE_DIR="$KV_SCRATCH/torchinductor-cache/$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$KV_SCRATCH/triton-cache/$SLURM_JOB_ID"

mkdir -p \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR"

echo "========================================"
echo "FULL STATIC CACHE COMPILE BENCHMARK"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
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

python -u src/bench_static_compile.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --contexts 2048 8192 12288 16384 32768 40895 \
    --new-tokens 64 \
    --repeats 5 \
    --eager-warmups 1 \
    --compile-warmups 3 \
    --out "results/raw/static_compile_qwen3_0.6b_${SLURM_JOB_ID}.csv"

echo "========================================"
echo "FINISHED: $(date)"
echo "========================================"
