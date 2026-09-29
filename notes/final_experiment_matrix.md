# Финальная экспериментальная матрица

Этот файл фиксирует последний набор экспериментов до просмотра их
результатов.

После завершения перечисленных ниже jobs новые экспериментальные sweep
не запускаются, кроме повторения конкретной конфигурации при технической
ошибке или подозрительном результате.

## Уже завершено

### FP16 KV Cache

- DynamicCache vs StaticCache на Qwen3-0.6B;
- long-context sweep до нативного предела ~40K;
- отдельные prefill/decode/TTFT/TPOT measurements;
- сравнение фактического KV storage с теоретической оценкой;
- StaticCache + torch.compile;
- диагностика Offloaded Cache correctness.

### Quantized KV Cache

Backend: HQQ.

Performance/memory scaling:

- Qwen3-0.6B;
- Qwen3-1.7B;
- Qwen3-4B;
- Qwen3-8B.

Основные caches:

- Dynamic FP16;
- HQQ INT4, q_group_size=64;
- HQQ INT2, q_group_size=64.

Quality scaling:

- Qwen3-0.6B;
- Qwen3-1.7B;
- Qwen3-4B;
- Qwen3-8B;
- Qwen3-14B.

Quality benchmark:

- controlled passkey retrieval;
- contexts 8K / 16K / 32K;
- needle positions 10% / 50% / 90%;
- Dynamic / INT4 / INT2.

### Дополнительные проверки

- INT8 group=64 на Qwen3-1.7B;
- INT4 group=16/64/128 на Qwen3-1.7B;
- INT3 отдельно диагностирован;
- INT3 direct HQQ quantize -> dequantize не работает
  в текущем HQQ stack для group_size 16/32/64/128;
- Quanto backend исключён из-за несовместимости kernels с V100.

## Последний блок A: quantization granularity

Модель: Qwen3-1.7B.

### Quality

Недостающие точки:

- INT2 group=16;
- INT2 group=32;
- INT2 group=128;
- INT4 group=32.

Уже существующие результаты будут объединены с ними:

- INT2 group=64;
- INT4 group=16;
- INT4 group=64;
- INT4 group=128;
- INT8 group=64;
- Dynamic.

Итоговая основная group-size матрица:

| Cache | g16 | g32 | g64 | g128 |
|---|---|---|---|---|
| INT2 | yes | yes | yes | yes |
| INT4 | yes | yes | yes | yes |

INT8 group=64 используется как near-lossless quantized reference.

### Performance

Для Qwen3-1.7B:

- Dynamic;
- INT8 g64;
- INT2 g16/g32/g64/g128;
- INT4 g16/g32/g64/g128.

Contexts:

- 8192;
- 16384;
- 32768;
- 40895.

Для каждой quantized configuration измеряются:

- physical KV storage;
- total peak GPU memory;
- prefill;
- TTFT;
- decode throughput;
- TPOT;
- end-to-end latency.

Цель: получить полный quality-memory-latency trade-off.

## Последний блок B: short-context quality control

Модель: Qwen3-0.6B.

Caches:

- Dynamic;
- INT4 g64;
- INT2 g64.

Contexts:

- 2048;
- 4096.

Positions:

- 0.1;
- 0.5;
- 0.9.

Seeds:

- 10.

Цель: проверить, присутствует ли деградация quantized KV Cache уже
при относительно коротком контексте, или она появляется только после
роста длины KV Cache.

## Явно не включается

### 14B performance

Qwen3-14B требует 2xV100 и device_map=balanced, тогда как performance
scaling 0.6B--8B измерялся на одной V100.

Добавление 14B в ту же performance curve одновременно изменило бы:

- model size;
- GPU count;
- model placement;
- inter-GPU communication;
- memory topology.

Поэтому 14B используется только для quality validation.

### Residual length ablation

residual_length=128 фиксируется во всех основных HQQ experiments.

Основной long prompt квантизируется уже при initial prefill.
При max_new_tokens=16 свежие decode states остаются внутри residual
window, поэтому benchmark главным образом исследует качество
квантизированного prompt KV Cache.

Изменение residual_length отвечало бы уже на дополнительный вопрос
о реквантизации свежих decode states.

### INT1

INT2 уже достигает quality floor на малых моделях.
INT1 не заполняет промежуток между двумя основными режимами.

### INT8 group-size sweep

INT8 g64 уже является near-lossless reference.
Group-size sweep проводится для INT2/INT4, где изменение granularity
реально влияет на качество.

### Модели крупнее 14B

Quality scaling уже покрывает:

0.6B -> 1.7B -> 4B -> 8B -> 14B.

Следующие модели были бы продолжением существующей тенденции,
а не закрытием отсутствующей точки текущей постановки.

## Stop criterion

После успешного завершения блоков A и B экспериментальная часть
исследования считается завершённой.

Разрешены только:

1. повтор конфигурации, которая завершилась технической ошибкой;
2. повтор явно аномальной точки для проверки воспроизводимости.

Новые размеры моделей, bit widths, group sizes, context grids и
benchmark families после этого не добавляются.
