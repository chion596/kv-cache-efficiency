# Ограничение Quanto QuantizedCache на V100

## Конфигурация

- GPU: NVIDIA Tesla V100-SXM2 32 GB
- compute capability: 7.0 (SM70)
- PyTorch: 2.14.0+cu126
- Transformers: 5.17.0
- optimum-quanto: 0.2.7

## Эксперимент

Проверялся `QuantizedCache` Transformers с backend `quanto`
для INT4/INT2 KV Cache.

DynamicCache успешно прошёл контрольный запуск.

Первый INT4 запуск завершился на этапе JIT-сборки CUDA extension
`quanto_cuda`.

## Причина

CUDA extension собирался с target `compute_70/sm_70`, соответствующим
Tesla V100.

При этом включённые в extension Marlin kernels используют инструкции,
которые требуют более новых архитектур GPU, в частности SM80.

В результате `ptxas` завершил сборку с ошибками вида:

`Feature '.m16n8k16' requires .target sm_80 or higher`

и

`Feature 'cp.async' requires .target sm_80 or higher`.

## Вывод

В текущей конфигурации кластера backend Quanto нельзя использовать
для Quantized KV Cache на доступной V100 без модификации самого
optimum-quanto.

Для дальнейшего эксперимента используется другой поддерживаемый
Transformers backend — HQQ.
