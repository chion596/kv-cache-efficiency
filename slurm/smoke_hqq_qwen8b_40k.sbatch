#!/bin/bash

#SBATCH --job-name=kv-8b-smoke
#SBATCH --partition=rocky
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a

#SBATCH --time=00:45:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

echo "========================================"
echo "QWEN3-8B 40K MEMORY SMOKE"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "========================================"

python - <<'PY'
import torch
import transformers
import hqq

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))
print("GPU memory GiB:",
      torch.cuda.get_device_properties(0).total_memory / 2**30)
print("hqq:", getattr(hqq, "__version__", "unknown"))
PY

python -u src/bench_quantized_prefill_decode.py \
    --model "$KV_SCRATCH/models/Qwen3-8B" \
    --contexts 40895 \
    --caches dynamic int4 int2 \
    --new-tokens 16 \
    --repeats 1 \
    --out "results/raw/smoke_hqq_qwen3_8b_40k_${SLURM_JOB_ID}.csv"

echo "========================================"
echo "FINISHED: $(date)"
echo "========================================"
