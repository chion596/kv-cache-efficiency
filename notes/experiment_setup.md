# Исходная постановка эксперимента

> Историческая заметка: фиксирует первоначальный план до перехода к финальной HQQ-матрице. Актуальная методика описана в `methodology_details.md`, итоговые результаты — в корневом `README.md`.

## Цель

Проверить, как стратегия KV Cache и длина контекста влияют на:

- GPU memory;
- скорость инференса;
- prefill и decode по отдельности.

## Начальная матрица

Первая модель:

```text
Qwen3-0.6B
```

Начальные cache strategies:

```text
DynamicCache
StaticCache
Offloaded Cache
```

Планировавшееся продолжение:

```text
Quantized Cache INT4
Quantized Cache INT2
Qwen3-1.7B
Qwen3-4B
```

Начальные длины контекста:

```text
512
2048
4096
8192
16384
```

## Метрики

В первой версии измерялись:

- время выполнения;
- tokens/s;
- peak allocated/reserved GPU memory;
- CPU RAM.

После pilot benchmark был расширен и начал отдельно измерять:

- prefill;
- TTFT;
- decode throughput;
- TPOT;
- теоретический размер KV Cache;
- фактический KV storage на GPU/CPU.

## Контролируемые параметры

При сравнении фиксировались:

- модель и GPU;
- FP16;
- длина output;
- способ построения input;
- версии библиотек;
- число повторов;
- warm-up.

## Первый запуск

```text
GPU: NVIDIA Tesla V100-SXM2 32 GB
Model: Qwen3-0.6B
Output: 64 tokens
Repeats: 3 after warm-up
```

Главное ограничение первой версии: total time смешивал prefill и decode. Поэтому pilot использовался только для проверки инфраструктуры, а финальные performance-выводы строились уже по раздельным измерениям.
