# `src/`: benchmark и analysis scripts

В этой папке лежит код экспериментов. Для понимания финального результата достаточно трёх файлов.

## Основной pipeline

### `bench_long_context_quality.py`

Основной quality benchmark.

Что делает:

- строит deterministic passkey-retrieval задачу;
- запускает DynamicCache или HQQ QuantizedCache;
- считает exact retrieval;
- считает teacher-forced top-1 и NLL;
- поддерживает `q_group_size`, `residual_length` и multi-GPU `device_map`.

Используется для model-size scaling, group-size ablation и short-context control.

### `bench_quantized_prefill_decode.py`

Основной performance/memory benchmark для HQQ.

Измеряет:

- prefill;
- decode throughput;
- end-to-end latency;
- peak allocated GPU memory;
- physical KV Cache storage.

Используется для model-size performance scaling и финального Qwen3-1.7B group-size ablation.

### `analyze_final.py`

Собирает committed raw CSV в:

```text
results/summary/final/
```

и строит:

```text
plots/final/
```

Основной способ воспроизвести финальные таблицы и графики из уже сохранённых результатов:

```bash
python src/analyze_final.py
```

Скрипт читает canonical raw CSV, сохранённые в репозитории. Их имена включают исходные Slurm job IDs.

## Baseline и вторичные эксперименты

- `bench_kv.py` — самый ранний pilot benchmark.
- `bench_prefill_decode.py` — FP16 Dynamic/Static benchmark с раздельными prefill/decode измерениями.
- `bench_static_compile.py` — StaticCache и `torch.compile`.
- `analyze.py` — baseline Dynamic/Static analysis; outputs идут в `results/summary/baseline/` и `plots/baseline/`.
- `analyze_static_compile.py` — анализ StaticCache + compile.
- `analyze_quantized_scaling.py` — ранний cross-model performance/memory analysis HQQ.
- `analyze_quality_scaling.py` — ранний cross-model quality analysis.
- `analyze_long_context_quality.py` — промежуточный анализ long-context quality.

Эти файлы сохраняются как воспроизводимая история исследования, но финальные выводы строятся через `analyze_final.py`.

## Correctness и diagnostics

- `check_cache_correctness.py` — correctness checks cache strategies.
- `check_quantized_cache.py` — проверка доступности quantized backends/configurations.
- `check_static_compile.py` — correctness диагностика StaticCache + compile.
- `diagnose_hqq_int3.py` — диагностика INT3 HQQ quantize/dequantize path.
- `diagnose_offloaded_cache.py` — диагностика нестабильности Offloaded Cache.

Diagnostic scripts не входят в основной quality/memory/latency comparison.

## Где запускать

GPU experiments запускались через Slurm jobs из:

```text
slurm/
```

Фиксированное окружение:

```text
requirements.lock.txt
```

Методические детали:

```text
notes/methodology_details.md
```

Финальная интерпретация результатов:

```text
README.md
```
