# HQQ Quantized KV Cache: Qwen3-0.6B

> Исторический HQQ performance/memory experiment. Для финального сравнения используются canonical таблицы из `results/summary/final/`.

## Конфигурация

```text
Model: Qwen3-0.6B
GPU: NVIDIA Tesla V100-SXM2 32 GB
PyTorch: 2.14.0+cu126
Transformers: 5.17.0
Backend: HQQ
Model dtype: FP16
Output: 64 tokens
Repeats: 5
Contexts: 2048–40895
```

Сравнивались DynamicCache FP16, HQQ INT4 и HQQ INT2.

## Физический размер KV Cache

Относительно FP16 DynamicCache:

```text
INT4: 28.125% storage -> 3.556x compression
INT2: 15.625% storage -> 6.4x compression
```

Экономия физического KV storage:

```text
INT4: 71.875%
INT2: 84.375%
```

Из измерений следует эффективная стоимость около:

```text
INT4: 4.5 bit/value
INT2: 2.5 bit/value
```

Разница с идеальными 4x/8x связана с quantization metadata.

## Total GPU memory

| Context | INT4 saving | INT2 saving |
|---:|---:|---:|
| 2048 | 9.7% | 11.6% |
| 8192 | 24.4% | 29.3% |
| 12288 | 29.4% | 35.2% |
| 16384 | 32.7% | 39.2% |
| 24576 | 36.8% | 44.2% |
| 32768 | 39.4% | 47.3% |
| 40895 | 41.1% | 49.3% |

На коротком контексте значительную часть памяти занимают веса модели; с ростом контекста доля KV Cache увеличивается, поэтому quantization сильнее влияет на total memory.

## Производительность

В этом эксперименте HQQ quantization не ускоряет decode:

```text
INT4 slowdown: примерно 19–36%
INT2 slowdown: примерно 25–37%
```

На context=40895:

```text
INT4 total-memory saving: 41.1%
INT4 end-to-end latency increase: ~69.7%

INT2 total-memory saving: 49.3%
INT2 end-to-end latency increase: ~71.7%
```

Физический KV storage на этой точке:

```text
INT4: ~1.229 GiB
INT2: ~0.683 GiB
```

## Корректность и ограничение

Повторы каждой пары context/cache были детерминированы по output hash, но quantized outputs отличались от Dynamic FP16. Само по себе это ожидаемо и не является quality metric.

Поэтому performance/memory experiment был дополнен отдельным retrieval benchmark.

Результат относится к Qwen3-0.6B, V100, HQQ, FP16 model weights, batch size 1 и greedy decoding. Для итоговых выводов о качестве и model scaling используются более поздние эксперименты.
