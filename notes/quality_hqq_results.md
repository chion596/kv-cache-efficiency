# HQQ quality: Qwen3-0.6B

> Ранний quality experiment. Финальное model-size и group-size сравнение находится в корневом `README.md` и `results/summary/final/`.

## Постановка

Controlled passkey retrieval:

```text
Contexts: 8192, 16384, 32768
Needle positions: 10%, 50%, 90%
Seeds: 10
Tasks per cache: 90
```

Сравнивались:

```text
DynamicCache FP16
HQQ INT4
HQQ INT2
```

Конфигурация quantized cache:

```text
q_group_size = 64
residual_length = 128
axis_key = 1
axis_value = 1
```

Метрики:

- exact-match retrieval;
- teacher-forced top-1;
- teacher-forced NLL.

## Общий результат

| Cache | Exact retrieval | TF top-1 | TF NLL |
|---|---:|---:|---:|
| Dynamic FP16 | 100.00% | 100.00% | 0.0155 |
| HQQ INT4 | 14.44% | 72.64% | 0.8541 |
| HQQ INT2 | 0.00% | 2.08% | 7.7843 |

Dynamic решил 90/90 задач. На той же постановке INT4 решил 13/90, INT2 — 0/90.

Разрыв между INT4 exact-match и TF top-1 означает, что многие отдельные target tokens предсказываются правильно, но ошибка хотя бы в одном месте ломает полный восьмизначный passkey.

## Длина контекста

Exact-match:

| Context | Dynamic | INT4 | INT2 |
|---:|---:|---:|---:|
| 8192 | 100.0% | 13.3% | 0.0% |
| 16384 | 100.0% | 13.3% | 0.0% |
| 32768 | 100.0% | 16.7% | 0.0% |

В диапазоне 8K–32K нет свидетельства монотонного ухудшения INT4 с ростом context length. Teacher-forced metrics также не показывают такого тренда.

Следовательно, этот эксперимент не поддерживает объяснение, в котором деградация вызвана только очень длинным KV Cache.

## Позиция needle

INT4 aggregate exact-match:

```text
10% -> 16.7%
50% -> 20.0%
90% ->  6.7%
```

Каждая context × position ячейка содержит только 10 задач, а teacher-forced metrics не дают устойчивого монотонного positional pattern. Поэтому отдельный positional effect по этим данным не утверждается.

## Что было проверено дальше

Финальная экспериментальная серия отдельно проверила:

- более короткие контексты 2K/4K;
- Qwen3 от 0.6B до 14B;
- влияние `q_group_size`.

`residual_length` в основной работе оставался фиксированным.
