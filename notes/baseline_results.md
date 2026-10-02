# Baseline: DynamicCache и StaticCache

> Исторический baseline на Qwen3-0.6B. Более плотный long-context sweep находится в `long_context_results.md`.

## Конфигурация

```text
GPU: NVIDIA Tesla V100-SXM2 32 GB
Model: Qwen3-0.6B
dtype: FP16
PyTorch: 2.14.0+cu126
Transformers: 5.17.0
Output: 64 tokens
Repeats: 5 after warm-up
Compilation: off
```

Контексты:

```text
512, 1024, 2048, 4096, 8192, 16384
```

## Decode throughput

| Context | Dynamic | Static |
|---:|---:|---:|
| 512 | 31.20 | 27.05 |
| 1024 | 31.42 | 26.92 |
| 2048 | 31.28 | 26.87 |
| 4096 | 31.30 | 26.89 |
| 8192 | 31.28 | 26.91 |
| 16384 | 22.53 | 18.03 |

До 8192 токенов DynamicCache держится около 31 tok/s. На 16384 throughput заметно падает.

StaticCache без compilation медленнее DynamicCache примерно на 13–14% до 8192 и примерно на 20% при 16384.

## Memory и корректность

StaticCache не дал экономии peak GPU memory относительно DynamicCache.

Для DynamicCache фактический FP16 KV storage практически совпал с теоретической оценкой.

Все пять повторов каждой конфигурации были воспроизводимы по output hash; Dynamic и Static также совпадали между собой.

## Что дал baseline

Baseline показал две вещи:

1. StaticCache без compilation не улучшает performance/memory в этой конфигурации.
2. Для локализации начала long-context slowdown нужна более плотная сетка между 8K и 16K.

Эта проверка была выполнена следующим long-context sweep.
