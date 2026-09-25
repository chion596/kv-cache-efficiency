#!/bin/bash

#SBATCH --job-name=kv-diag-main
#SBATCH --partition=test
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a

#SBATCH --time=00:20:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache-main

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

echo "========================================"
echo "TRANSFORMERS MAIN OFFLOAD TEST"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "PROJECT COMMIT: ${GIT_COMMIT:-unknown}"
echo "TRANSFORMERS COMMIT: ${HF_TRANSFORMERS_SHA:-unknown}"
echo "========================================"

python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("transformers path:", transformers.__file__)
print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "===== CORRECTNESS TEST ====="

python -u src/diagnose_offloaded_cache.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --attn sdpa \
    --contexts 1536 2048 4096 \
    --repeats 5 \
    --new-tokens 32

echo
echo "========================================"
echo "FINISHED: $(date)"
echo "========================================"
