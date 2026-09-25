#!/bin/bash

#SBATCH --job-name=kv-a100
#SBATCH --partition=test
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_e

#SBATCH --time=00:30:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

echo "============================================================"
echo "A100 OFFLOADED CACHE HARDWARE CONTROL"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "PROJECT COMMIT: ${GIT_COMMIT:-unknown}"
echo "MAIN TRANSFORMERS COMMIT: ${HF_TRANSFORMERS_SHA:-unknown}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

nvidia-smi

echo
echo "############################################################"
echo "# PART 1: TRANSFORMERS 5.17.0"
echo "############################################################"
echo

conda activate kvcache

python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM GiB:",
    torch.cuda.get_device_properties(0).total_memory / 2**30
)
PY

python -u src/diagnose_offloaded_cache.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --attn sdpa \
    --contexts 2048 4096 8192 \
    --repeats 5 \
    --new-tokens 32

echo
echo "############################################################"
echo "# PART 2: TRANSFORMERS MAIN"
echo "############################################################"
echo

conda activate kvcache-main

python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM GiB:",
    torch.cuda.get_device_properties(0).total_memory / 2**30
)
PY

python -u src/diagnose_offloaded_cache.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --attn sdpa \
    --contexts 2048 4096 8192 \
    --repeats 5 \
    --new-tokens 32

echo
echo "============================================================"
echo "FINISHED: $(date)"
echo "============================================================"
