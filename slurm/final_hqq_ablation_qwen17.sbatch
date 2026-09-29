#!/bin/bash

#SBATCH --job-name=kv-final-ab17
#SBATCH --partition=rocky
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a

#SBATCH --time=04:00:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

echo "============================================================"
echo "FINAL QWEN3-1.7B HQQ GRANULARITY ABLATION"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unknown}"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total \
    --format=csv,noheader

run_quality () {
    CACHE="$1"
    GROUP="$2"
    LABEL="$3"

    echo
    echo "============================================================"
    echo "QUALITY CACHE=$CACHE GROUP=$GROUP"
    echo "============================================================"

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
        "results/raw/final_quality_ablation_qwen3_1.7b_${LABEL}_${SLURM_JOB_ID}.csv"
}

echo
echo "============================================================"
echo "PART 1: MISSING QUALITY CELLS"
echo "============================================================"

run_quality int2 16  int2_g16
run_quality int2 32  int2_g32
run_quality int2 128 int2_g128
run_quality int4 32  int4_g32


run_perf () {
    CACHE="$1"
    GROUP="$2"
    LABEL="$3"

    echo
    echo "============================================================"
    echo "PERFORMANCE CACHE=$CACHE GROUP=$GROUP"
    echo "============================================================"

    python -u src/bench_quantized_prefill_decode.py \
        --model "$KV_SCRATCH/models/Qwen3-1.7B" \
        --contexts \
            8192 \
            16384 \
            32768 \
            40895 \
        --caches "$CACHE" \
        --q-group-size "$GROUP" \
        --residual-length 128 \
        --new-tokens 64 \
        --repeats 3 \
        --out \
        "results/raw/final_perf_ablation_qwen3_1.7b_${LABEL}_${SLURM_JOB_ID}.csv"
}

echo
echo "============================================================"
echo "PART 2: COMPLETE PERFORMANCE MATRIX"
echo "============================================================"

run_perf dynamic 64 dynamic

run_perf int8 64 int8_g64

run_perf int2 16  int2_g16
run_perf int2 32  int2_g32
run_perf int2 64  int2_g64
run_perf int2 128 int2_g128

run_perf int4 16  int4_g16
run_perf int4 32  int4_g32
run_perf int4 64  int4_g64
run_perf int4 128 int4_g128

echo
echo "============================================================"
echo "FINAL QWEN3-1.7B ABLATION FINISHED"
echo "DATE: $(date)"
echo "============================================================"
