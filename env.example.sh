# Пример настройки окружения для HPC-кластера НИУ ВШЭ.

# Создайте workspace:
# ws_allocate kv-cache 30

export KV_SCRATCH="$(ws_find kv-cache)"
export HF_HOME="$KV_SCRATCH/hf-cache"
export TMPDIR="$KV_SCRATCH/tmp"
export PIP_CACHE_DIR="$KV_SCRATCH/pip-cache"

mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR"
