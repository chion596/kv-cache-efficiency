#!/bin/bash

#SBATCH --job-name=kv-perf-ab17
#SBATCH --partition=rocky
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a

#SBATCH --time=01:30:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

echo "========================================"
echo "QWEN3-1.7B HQQ PERFORMANCE ABLATION"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "========================================"

run_config () {
    CACHE="$1"
    GROUP="$2"
    LABEL="$3"

    echo
    echo "========================================"
    echo "CACHE=$CACHE GROUP=$GROUP"
    echo "========================================"

    python -u src/bench_quantized_prefill_decode.py \
        --model "$KV_SCRATCH/models/Qwen3-1.7B" \
        --contexts \
            32768 \
            40895 \
        --caches "$CACHE" \
        --q-group-size "$GROUP" \
        --residual-length 128 \
        --new-tokens 64 \
        --repeats 3 \
        --out \
        "results/raw/perf_ablation_qwen3_1.7b_${LABEL}_${SLURM_JOB_ID}.csv"
}

run_config int8 64 int8_g64
run_config int4 16 int4_g16
run_config int4 64 int4_g64
run_config int4 128 int4_g128

echo
echo "========================================"
echo "FINISHED: $(date)"
echo "========================================"
