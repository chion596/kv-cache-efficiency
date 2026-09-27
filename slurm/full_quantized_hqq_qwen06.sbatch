#!/bin/bash

#SBATCH --job-name=kv-hqq-full
#SBATCH --partition=rocky
#SBATCH --account=proj_1782

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a

#SBATCH --time=01:00:00

#SBATCH --output=/home/nashilovskiy/kv-cache-research/logs/%x-%j.out
#SBATCH --error=/home/nashilovskiy/kv-cache-research/logs/%x-%j.err

set -euo pipefail

module purge
module load python
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

echo "========================================"
echo "FULL HQQ QUANTIZED KV CACHE BENCHMARK"
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
print("hqq:", getattr(hqq, "__version__", "unknown"))
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU capability:",
    torch.cuda.get_device_capability(0),
)
PY

python -u src/bench_quantized_prefill_decode.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --contexts \
        2048 \
        8192 \
        12288 \
        16384 \
        24576 \
        32768 \
        40895 \
    --caches \
        dynamic \
        int4 \
        int2 \
    --new-tokens 64 \
    --repeats 5 \
    --out "results/raw/full_quantized_hqq_qwen3_0.6b_${SLURM_JOB_ID}.csv"

echo "========================================"
echo "FINISHED: $(date)"
echo "========================================"
