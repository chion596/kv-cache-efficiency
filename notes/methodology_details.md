# Технические детали методики

## Quality benchmark

Скрипт:

```text
src/bench_long_context_quality.py
```

### Задача

В детерминированный synthetic filler вставляется уникальный восьмизначный passkey. После контекста модель должна вернуть только восемь цифр.

Needle вставляется примерно на:

```text
10%
50%
90%
```

длины body.

Prompts формируются как raw text/raw tokens:

- `apply_chat_template` не используется;
- отдельный Qwen3 thinking mode не включается.

Filler не содержит других восьмизначных кодов.

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

Teacher-forced decode metrics оценивают последующие target tokens; первый target token не используется как основной сигнал качества quantized cache.

### Размер выборки

Model-size experiment:

```text
Qwen3-0.6B: n=45
Qwen3-1.7B: n=45
Qwen3-4B:   n=45
Qwen3-8B:   n=45
Qwen3-14B:  n=45
```

Для основного cross-model comparison используется общий набор задач: 3 контекста × 3 позиции × seeds `0–4`. Исходный запуск Qwen3-0.6B содержит также seeds `5–9`, но эти дополнительные 45 задач не входят в RQ2 comparison. Полный 0.6B результат (`n=90`) сохраняется отдельно в `notes/quality_hqq_results.md`.

Group-size ablation:

```text
n=45 на конфигурацию
```

Short-context control:

```text
n=30 на cache/context
```

Для exact-match рассчитываются 95% интервалы Уилсона.

## Performance benchmark

Скрипт:

```text
src/bench_quantized_prefill_decode.py
```

Измеряются:

- prefill time;
- decode throughput;
- end-to-end time;
- peak allocated GPU memory;
- физический размер KV Cache.

### Peak GPU memory

Перед измерением вызывается:

```python
torch.cuda.reset_peak_memory_stats()
```

Peak memory считывается через:

```python
torch.cuda.max_memory_allocated()
```

### Physical KV storage

Размер cache считается по фактическим tensor storages.

Для DynamicCache учитываются K/V tensors.

Для HQQ QuantizedCache учитываются:

- quantized K/V;
- quantization metadata;
- residual FP16 K/V.

Повторно используемые storages дедуплицируются по data pointer.

### Повторы

Model-scaling performance:

```text
Qwen3-0.6B: 5
Qwen3-1.7B: 3
Qwen3-4B:   3
Qwen3-8B:   3
```

Final Qwen3-1.7B group-size ablation:

```text
3 повтора на context/configuration
```

В финальных таблицах сохраняются mean и sample standard deviation.

## HQQ configuration

Основная конфигурация:

```text
backend = hqq
axis_key = 1
axis_value = 1
q_group_size = 64
residual_length = 128
```

В group-size ablation на Qwen3-1.7B меняется только:

```text
q_group_size = 16, 32, 64, 128
```

для INT4 и INT2.

INT8 используется как reference при `q_group_size=64`.

`axis_key` и `axis_value` не входят в основную экспериментальную матрицу.

## Attention implementation

`attn_implementation` явно не задаётся. В проверенной основной конфигурации Qwen3-1.7B на NVIDIA V100-SXM2 32 GB, PyTorch 2.14.0+cu126 и Transformers 5.17.0 фактически выбрана SDPA:

```text
model.config._attn_implementation = "sdpa"
model.config._attn_implementation_internal = "sdpa"
```

Это значение проверено после загрузки модели тем же `AutoModelForCausalLM.from_pretrained(...)` path, который используется в benchmark.

## Canonical final output

Финальный анализ:

```text
src/analyze_final.py
```

Canonical таблицы:

```text
results/summary/final/
```

Для Qwen3-1.7B quality-memory-latency сравнения основной источник:

```text
memory_latency_tradeoff.csv
```

Для графиков memory/decode vs context:

```text
qwen17_final_context_performance.csv
```

## Ограничения

- Quality benchmark — controlled synthetic retrieval, а не широкий long-context benchmark.
- Основные performance-результаты относятся к NVIDIA V100 и batch size 1.
- Qwen3-14B использована только в quality scaling на двух V100.
- Интервалы Уилсона описывают uncertainty внутри benchmark trials и не доказывают перенос результатов на другие задачи или модели.
