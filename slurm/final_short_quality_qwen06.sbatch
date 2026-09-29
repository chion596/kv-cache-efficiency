#!/bin/bash

#SBATCH --job-name=kv-q06-short
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

echo "============================================================"
echo "FINAL QWEN3-0.6B SHORT-CONTEXT QUALITY CONTROL"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unknown}"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total \
    --format=csv,noheader

python -u src/bench_long_context_quality.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --contexts \
        2048 \
        4096 \
    --positions \
        0.1 \
        0.5 \
        0.9 \
    --caches \
        dynamic \
        int4 \
        int2 \
    --q-group-size 64 \
    --residual-length 128 \
    --num-seeds 10 \
    --max-new-tokens 16 \
    --out \
    "results/raw/final_quality_short_qwen3_0.6b_${SLURM_JOB_ID}.csv"

echo
echo "============================================================"
echo "FINAL SHORT-CONTEXT QUALITY CONTROL FINISHED"
echo "DATE: $(date)"
echo "============================================================"
