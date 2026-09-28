#!/bin/bash

#SBATCH --job-name=kv-quality-14b
#SBATCH --partition=rocky
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --constraint=type_a

#SBATCH --time=06:00:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

echo "========================================"
echo "QWEN3-14B LONG-CONTEXT QUALITY"
echo "JOB ID: $SLURM_JOB_ID"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "========================================"

python -u src/bench_long_context_quality.py \
    --model "$KV_SCRATCH/models/Qwen3-14B" \
    --contexts \
        8192 \
        16384 \
        32768 \
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
    --num-seeds 5 \
    --max-new-tokens 16 \
    --device-map balanced \
    --out \
    "results/raw/quality_hqq_qwen3_14b_${SLURM_JOB_ID}.csv"

echo "FINISHED: $(date)"
