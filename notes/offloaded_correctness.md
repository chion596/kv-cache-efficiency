# Проверка корректности Offloaded KV Cache

## Окружение

- Model: Qwen3-0.6B
- GPU: NVIDIA Tesla V100-SXM2 32 GB
- PyTorch: 2.14.0+cu126
- Transformers: 5.17.0
- Attention implementation: SDPA
- dtype: FP16
- decoding: greedy (`do_sample=False`)

## Наблюдение

При использовании стандартного `model.generate()` стратегии
DynamicCache и StaticCache дают идентичный и воспроизводимый output.

Для Offloaded Cache:

- context=512: output совпадает с DynamicCache;
- context=2048: output отличается и не воспроизводится между повторными запусками;
- context=8192: output отличается и не воспроизводится между повторными запусками.

Таким образом, эффект воспроизводится не только в собственном decode loop,
но и в стандартном Hugging Face `generate()`.

## Ограничение интерпретации

Пока нельзя утверждать, что это общий баг Transformers.

Необходимо проверить:

- другой attention backend;
- принудительную синхронизацию CUDA;
- Offloaded Static Cache;
- при необходимости другую версию Transformers и другой тип GPU.

До завершения этих проверок performance-результаты Offloaded Cache
не следует интерпретировать как корректное сравнение стратегий.

## Результаты расширенной диагностики

Проведено по 5 повторных greedy-запусков для нескольких длин контекста.

### SDPA

До context=1536:

- DynamicCache стабилен;
- Offloaded Cache стабилен;
- Offloaded Static Cache стабилен;
- все outputs совпадают.

При context=2048:

- DynamicCache остаётся детерминированным;
- Offloaded Cache: 5 различных outputs из 5 запусков;
- Offloaded Static Cache также становится нестабильным.

При context=4096:

- обе offloaded-стратегии нестабильны во всех повторах.

### Eager attention

Проблема проявляется позже:

- context <= 2048: offloading работает корректно;
- context=4096: Offloaded Cache и Offloaded Static Cache становятся нестабильными.

Это показывает, что порог возникновения проблемы зависит от attention backend.

### CUDA_LAUNCH_BLOCKING=1

При SDPA и принудительной синхронизации CUDA:

- context=512 — корректно;
- context=1024 — корректно;
- context=1536 — корректно;
- context=2048 — корректно;
- context=4096 — корректно.

Во всех случаях Offloaded Cache и Offloaded Static Cache дают тот же output,
что и DynamicCache, во всех пяти повторах.

## Предварительная интерпретация

Результат указывает на проблему, связанную с асинхронным выполнением или
синхронизацией при CPU↔GPU offloading.

Это пока не доказывает конкретный race condition внутри Transformers:
необходимо проверить другую версию Transformers и, при необходимости,
другой GPU / PyTorch.

Важно: CUDA_LAUNCH_BLOCKING=1 нельзя использовать как обычный performance
benchmark, поскольку он намеренно меняет режим исполнения CUDA и способен
существенно ухудшить производительность.

## Проверка на Transformers main

Дополнительно проведена проверка на development-версии Transformers:

- Transformers: 5.18.0.dev0
- commit: 5e4d6304de5536bc808187e5951f5e8794211229
- PyTorch: 2.14.0+cu126
- CUDA runtime PyTorch: 12.6
- GPU: NVIDIA Tesla V100-SXM2 32 GB
- attention backend: SDPA
- остальные параметры эксперимента сохранены.

### Результаты

При context=1536:

- DynamicCache стабилен;
- Offloaded Cache стабилен;
- Offloaded Static Cache стабилен;
- outputs совпадают.

При context=2048:

- все три стратегии дают одинаковый output;
- каждый вариант воспроизводим во всех 5 повторах.

При context=4096:

- DynamicCache остаётся детерминированным;
- Offloaded Cache даёт 5 различных outputs из 5 запусков;
- Offloaded Static Cache даёт 5 различных outputs из 5 запусков;
- offloaded outputs не совпадают с DynamicCache.

### Интерпретация

Переход с Transformers 5.17.0 на протестированный commit main
не устраняет проблему полностью.

На протестированных точках проблема при SDPA перестала проявляться
на context=2048, но сохраняется на context=4096.

Это не следует интерпретировать как точный фиксированный порог:
при вероятной проблеме синхронизации момент возникновения ошибки
может зависеть от timing, версии библиотек, attention backend и GPU.

Ранее CUDA_LAUNCH_BLOCKING=1 полностью устранял расхождение вплоть
до context=4096, что является дополнительным свидетельством связи
проблемы с асинхронным выполнением.
