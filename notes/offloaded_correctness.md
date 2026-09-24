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
