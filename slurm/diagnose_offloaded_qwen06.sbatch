#!/bin/bash

#SBATCH --job-name=kv-diagnose
#SBATCH --partition=test
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a

#SBATCH --time=00:30:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

echo "======================================="
echo "OFFLOADED CACHE DIAGNOSTICS"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "======================================="

echo
echo "===== SDPA ====="

python -u src/diagnose_offloaded_cache.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --attn sdpa

echo
echo "===== EAGER ====="

python -u src/diagnose_offloaded_cache.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --attn eager

echo
echo "===== SDPA + CUDA_LAUNCH_BLOCKING ====="

CUDA_LAUNCH_BLOCKING=1 \
python -u src/diagnose_offloaded_cache.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --attn sdpa
