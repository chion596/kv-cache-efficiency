# StaticCache + torch.compile

## Конфигурация

```text
GPU: NVIDIA Tesla V100-SXM2 32 GB
Model: Qwen3-0.6B
dtype: FP16
PyTorch: 2.14.0+cu126
Transformers: 5.17.0
Output: 64 tokens
Repeats: 5 after warm-up
Compile backend: inductor
Compile mode: reduce-overhead
```

Сравнивались:

- DynamicCache eager;
- StaticCache eager;
- StaticCache с automatic compilation в `generate()`.

Prefill остаётся eager; compiled forward используется для iterative decode.

## Корректность

Во всех конфигурациях:

- каждый режим воспроизводим;
- Static eager совпадает с Dynamic по output;
- Static compiled совпадает с Dynamic;
- graph breaks не наблюдались.

## Cold compilation

Первый compiled warm-up дорогой:

```text
context=2048: 89.98 s
context=8192: 52.09 s
```

После 8192 число compiled graphs остаётся равным двум до 40895.

Cold compile time не включается в steady-state сравнение.

## Steady-state end-to-end latency

| Context | Dynamic eager | Static eager | Static compiled |
|---:|---:|---:|---:|
| 2048 | 2.149 s | 2.472 s | 0.829 s |
| 8192 | 2.486 s | 2.828 s | 2.213 s |
| 12288 | 3.080 s | 3.645 s | 3.310 s |
| 16384 | 4.159 s | 4.934 s | 4.495 s |
| 32768 | 9.795 s | 11.237 s | 10.623 s |
| 40895 | 13.443 s | 15.229 s | 14.523 s |

Относительно Static eager compilation уменьшает latency на:

```text
2048:  66.5%
8192:  21.7%
12288:  9.2%
16384:  8.9%
32768:  5.5%
40895:  4.6%
```

## Интерпретация

Static compiled быстрее Dynamic при 2048 и 8192 токенах, но уже медленнее при 12288. По дискретной сетке crossover находится между 8192 и 12288; точный порог не определяется.

С ростом контекста доля eager prefill увеличивается, поэтому эффект compilation на end-to-end latency уменьшается.

Static compiled и Static eager имеют практически одинаковый peak GPU memory: compilation меняет compute performance, но не даёт дополнительной экономии KV storage.

## Вывод

- StaticCache без compilation медленнее DynamicCache.
- `torch.compile` сильно помогает на коротких контекстах.
- На длинных контекстах DynamicCache снова быстрее.
- Для практической оценки compiled режима необходимо отдельно учитывать большой cold-start cost.
