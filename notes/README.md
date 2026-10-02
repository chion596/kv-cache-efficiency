# Notes

В этой папке собраны методические заметки и промежуточные результаты экспериментов.

## Актуальные документы

- `literature.md` — related work, связь с KIVI и механизмами HQQ QuantizedCache.
- `methodology_details.md` — точная постановка quality/performance benchmark и правила статистического отчёта.

Финальные выводы и основные числа находятся в корневом `README.md`.

Canonical агрегированные результаты:

```text
results/summary/final/
```

Основные файлы:

```text
memory_latency_tradeoff.csv
qwen17_final_context_performance.csv
model_scaling_quality.csv
group_size_quality.csv
group_size_performance.csv
short_context_control.csv
```

## История экспериментов

Остальные файлы фиксируют отдельные этапы исследования:

- `experiment_setup.md` — исходная постановка;
- `pilot_results.md` — первый pilot;
- `baseline_results.md` — DynamicCache/StaticCache baseline;
- `long_context_results.md` — long-context sweep;
- `static_compile_results.md` — StaticCache + `torch.compile`;
- `quantized_hqq_results.md` — ранний performance/memory HQQ experiment;
- `quality_hqq_results.md` — ранний quality experiment на Qwen3-0.6B;
- `offloaded_correctness.md` — диагностика Offloaded Cache;
- `quanto_v100_limitation.md` — ограничение Quanto на V100;
- `final_experiment_matrix.md` — зафиксированная перед финальными запусками экспериментальная матрица.

Эти файлы полезны как журнал хода исследования, но могут содержать ранние или промежуточные числа. Для финального сравнения следует использовать `README.md` и `results/summary/final/`.
