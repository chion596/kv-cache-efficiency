#!/bin/bash

#SBATCH --job-name=kv-qwen06
#SBATCH --partition=rocky
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a

#SBATCH --time=02:00:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh

cd /home/nashilovskiy/kv-cache-research

echo "================================================"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "================================================"

nvidia-smi

python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM GB:",
        torch.cuda.get_device_properties(0).total_memory / 2**30,
    )
PY

python -u src/bench_kv.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --contexts 512 2048 4096 8192 16384 \
    --caches dynamic static offloaded \
    --new-tokens 64 \
    --repeats 3 \
    --out "results/qwen3_0.6b_${SLURM_JOB_ID}.csv"

echo "================================================"
echo "FINISHED: $(date)"
echo "================================================"
