# `results/`: сырые и агрегированные результаты

## Финальные результаты

Canonical таблицы, на которых построен корневой `README.md`:

```text
results/summary/final/
```

Основные файлы:

```text
memory_latency_tradeoff.csv
qwen17_final_context_performance.csv
model_scaling_performance.csv
model_scaling_quality.csv
group_size_performance.csv
group_size_quality.csv
short_context_control.csv
```

Они собираются скриптом:

```bash
python src/analyze_final.py
```

## Raw CSV

```text
results/raw/
```

Здесь сохранены исходные CSV конкретных Slurm-запусков.

Имена файлов содержат исходный `$SLURM_JOB_ID`, поэтому они одновременно служат связью между experiment configuration и результатом запуска.

`src/analyze_final.py` явно указывает, какие raw CSV считаются canonical inputs финального анализа.

## Исторические summaries

Остальные подпапки `results/summary/` относятся к промежуточным этапам исследования:

- `baseline/` — DynamicCache vs StaticCache baseline;
- `long_context/` — long-context FP16 experiment;
- `quality_hqq/` — ранний quality experiment;
- `quality_scaling/` — промежуточный cross-model quality analysis;
- `quantized_hqq*/` — ранние HQQ performance/memory summaries;
- `quantized_scaling/` — cross-model performance/memory analysis;
- `static_compile/` — StaticCache + `torch.compile`.

Для итоговых чисел следует использовать только `results/summary/final/`, если явно не исследуется история экспериментов.
