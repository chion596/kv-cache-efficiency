#!/bin/bash

#SBATCH --job-name=kv-ds-full
#SBATCH --partition=rocky
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a

#SBATCH --time=00:45:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

echo "========================================"
echo "FULL DYNAMIC / STATIC BENCHMARK"
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
PY

python -u src/bench_prefill_decode.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --contexts 512 1024 2048 4096 8192 16384 \
    --caches dynamic static \
    --new-tokens 64 \
    --repeats 5 \
    --out "results/raw/full_dynamic_static_qwen3_0.6b_${SLURM_JOB_ID}.csv"

echo "========================================"
echo "FINISHED: $(date)"
echo "========================================"
