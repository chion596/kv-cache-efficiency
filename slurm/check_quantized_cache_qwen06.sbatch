#!/bin/bash

#SBATCH --job-name=kv-quant
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
conda activate kvcache

source /home/nashilovskiy/kv-cache-research/env.sh
cd /home/nashilovskiy/kv-cache-research

echo "========================================"
echo "QUANTIZED KV CACHE SMOKE TEST"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "GIT COMMIT: ${GIT_COMMIT:-unknown}"
echo "========================================"

python - <<'PY'
import torch
import transformers
import optimum.quanto

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("optimum.quanto: import OK")
PY

python -u src/check_quantized_cache.py \
    --model "$KV_SCRATCH/models/Qwen3-0.6B" \
    --context 2048 \
    --new-tokens 32 \
    --repeats 3

echo "========================================"
echo "FINISHED: $(date)"
echo "========================================"
