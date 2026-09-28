#!/bin/bash

#SBATCH --job-name=kv-ablate-17
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

echo "========================================"
echo "QWEN3-1.7B HQQ ABLATION"
echo "JOB ID: $SLURM_JOB_ID"
echo "DATE: $(date)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "========================================"

run_quality () {
    CACHE="$1"
    GROUP="$2"
    LABEL="$3"

    echo
    echo "========================================"
    echo "CACHE=$CACHE GROUP=$GROUP"
    echo "========================================"

    python -u src/bench_long_context_quality.py \
        --model "$KV_SCRATCH/models/Qwen3-1.7B" \
        --contexts \
            8192 \
            16384 \
            32768 \
        --positions \
            0.1 \
            0.5 \
            0.9 \
        --caches "$CACHE" \
        --q-group-size "$GROUP" \
        --residual-length 128 \
        --num-seeds 5 \
        --max-new-tokens 16 \
        --out \
        "results/raw/quality_ablation_qwen3_1.7b_${LABEL}_${SLURM_JOB_ID}.csv"
}

run_quality int3 64 int3_g64
run_quality int8 64 int8_g64
run_quality int4 16 int4_g16
run_quality int4 128 int4_g128

echo "========================================"
echo "FINISHED: $(date)"
echo "========================================"
