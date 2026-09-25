#!/bin/bash

#SBATCH --job-name=kv-long-ds
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
echo "LONG CONTEXT DYNAMIC / STATIC BENCHMARK"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "========================================"

python - <<'PY'
import torch
import transformers
from transformers import AutoConfig

model_path = "/scratch/ws/nashilovskiy-kv-cache/models/Qwen3-0.6B"
cfg = AutoConfig.from_pretrained(
    model_path,
    local_files_only=True,
)

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("max_position_embeddings:", cfg.max_position_embeddings)

if cfg.max_position_embeddings < 40960:
    raise RuntimeError(
        f"Unexpected max_position_embeddings="
        f"{cfg.max_position_embeddings}"
    )
PY

python -u src/bench_prefill_decode.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --contexts 8192 10240 12288 14336 16384 20480 24576 32768 40895 \
    --caches dynamic static \
    --new-tokens 64 \
    --repeats 5 \
    --out "results/raw/long_context_dynamic_static_qwen3_0.6b_${SLURM_JOB_ID}.csv"

echo "========================================"
echo "FINISHED: $(date)"
echo "========================================"
