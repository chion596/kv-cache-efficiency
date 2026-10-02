# Технические детали экспериментальной методики

Эта заметка фиксирует детали, которые легко потерять при сокращении README.

## 1. Quality benchmark

Основной quality benchmark реализован в:

```text
src/bench_long_context_quality.py
```

### Формат входа

Используется raw text / raw token sequence.

`apply_chat_template` не используется.

Специальный chat-формат и отдельный Qwen3 thinking mode
в benchmark не включаются.

### Filler

Контекст заполняется детерминированным synthetic natural-ish текстом.

Пример структуры filler:

```text
Archive entry ...
The ... near the ... was reviewed under section ...
The routine note concerned ... and contained no special instructions.
```

Filler специально не содержит восьмизначных кодов,
чтобы secret passkey оставался уникальной retrieval-целью.

### Passkey

Для каждой комбинации:

```text
seed
context length
position
```

генерируется детерминированный случайный восьмизначный код.

Needle имеет форму:

```text
IMPORTANT MEMORY RECORD
The secret access code is XXXXXXXX.
Remember this exact eight-digit code.
END IMPORTANT MEMORY RECORD
```

После контекста добавляется прямой вопрос:

```text
What is the secret access code from the IMPORTANT MEMORY RECORD?
Reply with only the eight digits of the code, with no spaces or punctuation.
```

### Position

Needle вставляется примерно на:

```text
10%
50%
90%
```

длины body.

### Метрики

Основная:

```text
normalized_exact_match
```

Дополнительные:

```text
tf_top1_decode_pct
tf_nll_decode
```

Teacher-forced decode metrics не используют первый target token
как основной сигнал quantized-cache quality.

---

## 2. Размер выборки quality

Model-size experiment:

```text
Qwen3-0.6B: n = 90
Qwen3-1.7B: n = 45
Qwen3-4B:   n = 45
Qwen3-8B:   n = 45
Qwen3-14B:  n = 45
```

Group-size ablation:

```text
n = 45 на конфигурацию
```

Short-context control:

```text
n = 30 на cache/context configuration
```

Для exact retrieval финальный анализ использует
95% Wilson confidence intervals.

---

## 3. Performance benchmark

Основной benchmark реализован в:

```text
src/bench_quantized_prefill_decode.py
```

### Peak GPU memory

Перед измерением используется:

```python
torch.cuda.reset_peak_memory_stats()
```

Peak allocated GPU memory считывается через:

```python
torch.cuda.max_memory_allocated()
```

### Physical KV storage

Размер cache считается отдельно по реальным tensor storages.

Для DynamicCache учитываются:

```text
keys
values
```

Для HQQ QuantizedCache также учитываются:

```text
_quantized_keys
_quantized_values
quantization metadata
residual FP16 cache
```

Повторно используемые storages дедуплицируются по data pointer.

### Повторы

Model-scaling performance:

```text
Qwen3-0.6B: 5 repeats
Qwen3-1.7B: 3 repeats
Qwen3-4B:   3 repeats
Qwen3-8B:   3 repeats
```

Final Qwen3-1.7B group-size ablation:

```text
3 repeats на context/configuration
```

Для финального анализа сохраняются mean и sample standard deviation.

### Attention implementation

Benchmark явно не задаёт `attn_implementation`.

Поэтому в отчёте нельзя утверждать,
что была принудительно выбрана конкретная реализация attention.

Использовалась реализация, автоматически выбранная
Transformers/PyTorch для данного software/hardware stack.

---

## 4. Конфигурация QuantizedCache

В основной серии экспериментов использовался HQQ backend:

```text
backend = hqq
axis_key = 1
axis_value = 1
q_group_size = 64
residual_length = 128
```

В group-size ablation на Qwen3-1.7B менялся только:

```text
q_group_size = 16, 32, 64, 128
```

для INT4 и INT2.

INT8 использовался как reference configuration при:

```text
q_group_size = 64
```

Изменение `axis_key` и `axis_value` не входило в основную
экспериментальную матрицу. Это оставлено как возможное направление
отдельного исследования асимметричной квантизации Keys и Values.

---

## 5. Что считается canonical final output

Финальные агрегированные таблицы создаются:

```text
src/analyze_final.py
```

и сохраняются в:

```text
results/summary/final/
```

Основные файлы:

```text
model_scaling_quality.csv
model_scaling_performance.csv
group_size_quality.csv
group_size_performance.csv
short_context_control.csv
memory_latency_tradeoff.csv
```

Для финального Qwen3-1.7B quality-memory-latency сравнения
canonical источником чисел является:

```text
results/summary/final/memory_latency_tradeoff.csv
```

Именно из него должны браться итоговые значения
для Dynamic, INT8, INT4 и INT2 при контексте 40895.

---

## 6. Statistical reporting

Для бинарной метрики retrieval exact-match указываются:

```text
successes / n
accuracy, %
95% Wilson confidence interval
```

Для performance указываются:

```text
mean
sample standard deviation
n
```

Это позволяет отделить реальные различия между конфигурациями
от разброса между повторами.

---

## 7. Ограничения методики

Quality benchmark является контролируемой synthetic retrieval-задачей
и не заменяет широкий long-context benchmark.

Основные performance-результаты относятся к NVIDIA V100
и конкретному software stack.

Qwen3-14B использовалась только в quality scaling,
поскольку для неё потребовалось две V100,
и её performance нельзя напрямую сравнивать с 1-GPU результатами
для моделей 0.6B–8B.
