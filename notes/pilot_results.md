# Pilot: Dynamic, Static и Offloaded Cache

> Исторический pilot. Для финальных выводов используются более поздние benchmark и canonical таблицы из `results/summary/final/`.

## Конфигурация

```text
Model: Qwen3-0.6B
GPU: NVIDIA Tesla V100-SXM2 32 GB
dtype: FP16
Output: 32 tokens
Repeats: 1 after warm-up
Contexts: 512, 2048, 8192
```

Сравнивались:

- DynamicCache;
- StaticCache;
- Offloaded Cache.

## Наблюдения

DynamicCache был самым быстрым из трёх режимов.

Peak GPU memory DynamicCache:

| Context | Memory |
|---:|---:|
| 512 | 1.205 GB |
| 2048 | 1.389 GB |
| 8192 | 2.201 GB |

StaticCache без compilation имел близкое memory footprint, но более низкую скорость.

Offloaded Cache при 8192 токенах уменьшил GPU memory:

```text
Dynamic:   2.201 GB
Offloaded: 1.357 GB
```

но снизил throughput:

```text
Dynamic:   21.949 tok/s
Offloaded:  6.857 tok/s
```

## Ограничения

Pilot содержит только один measured run на конфигурацию и использует время, в котором prefill и decode ещё не разделены.

Поэтому числа этого файла не используются как финальный performance result. Его роль — подтвердить работоспособность benchmark pipeline и обозначить memory/latency trade-off для дальнейшей проверки.
