# Quanto QuantizedCache на V100

## Конфигурация

```text
GPU: NVIDIA Tesla V100-SXM2 32 GB
Compute capability: 7.0 (SM70)
PyTorch: 2.14.0+cu126
Transformers: 5.17.0
optimum-quanto: 0.2.7
```

Проверялся Transformers `QuantizedCache` с backend `quanto`.

DynamicCache прошёл control run, но первый INT4 запуск остановился при JIT-компиляции CUDA extension `quanto_cuda`.

## Причина

Extension собирался под:

```text
compute_70 / sm_70
```

но включённые Marlin kernels используют инструкции, требующие SM80 или новее.

`ptxas` сообщил, в частности:

```text
Feature '.m16n8k16' requires .target sm_80 or higher
Feature 'cp.async' requires .target sm_80 or higher
```

## Вывод

В протестированном software stack Quanto backend не удалось использовать для Quantized KV Cache на V100 без модификации `optimum-quanto`.

Поэтому основной quantized-cache experiment был продолжен с HQQ backend.

Этот результат описывает ограничение конкретного V100/software path и не является утверждением о Quanto на поддерживаемых новых GPU.
