# Offloaded KV Cache: проверка корректности

## Основная конфигурация

```text
Model: Qwen3-0.6B
GPU: NVIDIA Tesla V100-SXM2 32 GB
PyTorch: 2.14.0+cu126
Transformers: 5.17.0
Attention: SDPA
dtype: FP16
Decoding: greedy
```

Проверялись DynamicCache, Offloaded Cache и Offloaded Static Cache.

## Исходное наблюдение

В стандартном `model.generate()`:

```text
context=512:
offloaded output совпадает с Dynamic

context=2048:
offloaded output расходится с Dynamic
и меняется между повторными запусками

context=8192:
наблюдается та же нестабильность
```

DynamicCache и StaticCache при этом оставались детерминированными.

## Диагностика

Проведено по 5 greedy-запусков на конфигурацию.

### SDPA, Transformers 5.17.0

До 1536 токенов все стратегии стабильны и совпадают.

При 2048:

```text
Dynamic: stable
Offloaded: 5 разных outputs из 5
Offloaded Static: unstable
```

При 4096 обе offloaded-стратегии также нестабильны.

### Eager attention

```text
context <= 2048: корректно
context = 4096: offloaded-стратегии нестабильны
```

Момент появления эффекта зависит от attention backend.

### `CUDA_LAUNCH_BLOCKING=1`

При SDPA и принудительной CUDA synchronization все проверенные точки:

```text
512, 1024, 1536, 2048, 4096
```

дали одинаковый с DynamicCache output во всех пяти повторах.

Это связывает наблюдение с asynchronous execution/synchronization, но не доказывает конкретный race condition внутри Transformers.

`CUDA_LAUNCH_BLOCKING=1` нельзя использовать для обычного performance benchmark, потому что он меняет режим CUDA execution.

## Проверка development-версии Transformers

Дополнительная конфигурация:

```text
Transformers: 5.18.0.dev0
commit: 5e4d6304de5536bc808187e5951f5e8794211229
PyTorch: 2.14.0+cu126
CUDA runtime: 12.6
GPU: V100-SXM2 32 GB
Attention: SDPA
```

Результат:

```text
1536: все стратегии стабильны
2048: все стратегии стабильны
4096: обе offloaded-стратегии дают 5 разных outputs из 5
```

Переход на протестированный development commit не устранил проблему полностью.

## Ограничение

Все проверки выполнены на V100. A100/H100/H200 были недоступны этому аккаунту по политике Slurm, поэтому результат нельзя автоматически переносить на другие поколения GPU.

Из-за обнаруженной нестабильности Offloaded Cache не используется как основной performance baseline финальной работы.
