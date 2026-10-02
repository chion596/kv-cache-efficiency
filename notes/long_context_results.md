# Long-context benchmark: Qwen3-0.6B

## Конфигурация

```text
GPU: NVIDIA Tesla V100-SXM2 32 GB
Model: Qwen3-0.6B
dtype: FP16
PyTorch: 2.14.0+cu126
Transformers: 5.17.0
max_position_embeddings: 40960
Output: 64 tokens
Repeats: 5 after warm-up
Compilation: off
```

Сравнивались DynamicCache и StaticCache.

Контексты:

```text
8192, 10240, 12288, 14336, 16384,
20480, 24576, 32768, 40895
```

Все 90 измерений завершились без ошибок.

## Decode throughput

| Context | Dynamic | Static |
|---:|---:|---:|
| 8192 | 32.06 | 28.03 |
| 10240 | 31.84 | 27.41 |
| 12288 | 28.34 | 23.24 |
| 14336 | 25.13 | 20.31 |
| 16384 | 22.55 | 18.08 |
| 20480 | 18.74 | 14.95 |
| 24576 | 16.05 | 12.71 |
| 32768 | 12.46 | 9.80 |
| 40895 | 10.19 | 8.02 |

Первое выраженное снижение DynamicCache видно между 10240 и 12288 токенами. Это область начала slowdown, а не точный порог.

StaticCache без compilation медленнее DynamicCache на всём диапазоне; после начала long-context slowdown штраф составляет примерно 20–21%.

## Prefill

DynamicCache:

| Context | Prefill |
|---:|---:|
| 8192 | 0.413 s |
| 16384 | 1.276 s |
| 24576 | 2.662 s |
| 32768 | 4.578 s |
| 40895 | 7.063 s |

Prefill растёт быстрее длины контекста. По этим измерениям не делается вывод о точной асимптотике.

## KV Cache memory

DynamicCache совпадает с теоретической FP16-оценкой:

| Context | KV Cache |
|---:|---:|
| 8192 | 0.8750 GiB |
| 16384 | 1.7500 GiB |
| 24576 | 2.6250 GiB |
| 32768 | 3.5000 GiB |
| 40895 | 4.3681 GiB |

StaticCache немного больше из-за предварительного выделения до `max_cache_len`; при 40895 токенах он занимает 4.3750 GiB.

Peak GPU memory при 40895:

```text
Dynamic: 6.529 GiB
Static:  6.531 GiB
```

## Корректность и вывод

Все пять повторов каждой стратегии/длины дали одинаковый output hash.

Результат baseline уточняется: slowdown начинается уже после примерно 10K токенов. Для дальнейшего исследования memory pressure полезнее было перейти к quantized/offloaded cache и более крупным моделям, а не выходить за native context Qwen3-0.6B.
