#!/bin/bash

#SBATCH --job-name=kv-correct
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
echo "KV CACHE CORRECTNESS CHECK"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "======================================="

python -u src/check_cache_correctness.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --contexts 512 2048 8192 \
    --new-tokens 32
