# Финальная экспериментальная матрица

> Исторический документ: эта матрица была зафиксирована **до** последних запусков как stop criterion. Перечисленные финальные блоки впоследствии были выполнены. Итоговые числа находятся в корневом `README.md` и `results/summary/final/`.

## Уже выполненная база

### FP16 KV Cache

- DynamicCache vs StaticCache на Qwen3-0.6B;
- long-context sweep до ~40K;
- отдельные prefill/decode/TTFT/TPOT;
- сравнение theoretical и physical KV storage;
- StaticCache + `torch.compile`;
- диагностика Offloaded Cache.

### HQQ Quantized KV Cache

Performance/memory scaling:

```text
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
Qwen3-8B
```

Основные caches:

```text
Dynamic FP16
HQQ INT4 g64
HQQ INT2 g64
```

Quality scaling:

```text
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
Qwen3-8B
Qwen3-14B
```

Quality benchmark:

```text
contexts: 8K / 16K / 32K
needle positions: 10% / 50% / 90%
caches: Dynamic / INT4 / INT2
```

Дополнительные проверки до финального блока:

- INT8 g64 на Qwen3-1.7B;
- INT4 g16/g64/g128;
- диагностика INT3;
- Quanto исключён из-за V100 kernel incompatibility.

INT3 direct HQQ quantize → dequantize не работал в текущем HQQ stack для `group_size=16/32/64/128`.

## Финальный блок A: quantization granularity

Модель:

```text
Qwen3-1.7B
```

Group-size matrix:

| Cache | g16 | g32 | g64 | g128 |
|---|---|---|---|---|
| INT2 | yes | yes | yes | yes |
| INT4 | yes | yes | yes | yes |

INT8 g64 использовался как reference.

Performance contexts:

```text
8192
16384
32768
40895
```

На каждой конфигурации измерялись:

- physical KV storage;
- total peak GPU memory;
- prefill;
- TTFT;
- decode throughput;
- TPOT;
- end-to-end latency.

Цель блока — совместить quality, memory и latency в одном trade-off.

## Финальный блок B: short-context control

```text
Model: Qwen3-0.6B
Caches: Dynamic / INT4 g64 / INT2 g64
Contexts: 2048 / 4096
Positions: 0.1 / 0.5 / 0.9
Seeds: 10
```

Цель — проверить, появляется ли деградация quantized cache уже на коротком контексте.

## Что сознательно не добавлялось

### 14B performance

Qwen3-14B требует 2×V100 и `device_map=balanced`, тогда как 0.6B–8B performance scaling измерялся на одной V100. Добавление 14B одновременно меняло бы model size, GPU count, placement, communication и memory topology.

Поэтому 14B использовалась только для quality validation.

### Residual-length ablation

Во всех основных HQQ experiments:

```text
residual_length = 128
```

При `max_new_tokens=16` свежие decode states остаются внутри residual window. Изменение `residual_length` отвечало бы на отдельный вопрос и не входило в основную матрицу.

### INT1

INT2 уже давал quality floor на малых моделях, поэтому INT1 не закрывал важный промежуток основной постановки.

### INT8 group-size sweep

INT8 g64 использовался как near-lossless quantized reference. Group-size sweep был сфокусирован на INT2/INT4, где granularity заметно влияла на качество.

### Модели крупнее 14B

Quality scaling уже покрывал:

```text
0.6B -> 1.7B -> 4B -> 8B -> 14B
```

Дальнейшее расширение model-size grid не было необходимо для закрытия исходных research questions.

## Stop criterion

После блоков A и B новые model sizes, bit widths, group sizes, context grids и benchmark families не добавлялись.

Допускались только:

1. повтор технически упавшей конфигурации;
2. повтор явно аномальной точки для проверки воспроизводимости.
