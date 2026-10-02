# `slurm/`: jobs для кластера НИУ ВШЭ

Jobs в этой папке фиксируют параметры запусков на кластере. Они cluster-specific: содержат HSE account/partition, абсолютные пути `/home/nashilovskiy/...` и используют окружение `kvcache`.

## Jobs, связанные с финальными результатами

### Model-size performance scaling

```text
full_quantized_hqq_qwen06.sbatch
full_quantized_hqq_qwen17.sbatch
full_quantized_hqq_qwen4.sbatch
full_quantized_hqq_qwen8b.sbatch
```

Они запускают HQQ performance/memory benchmark для Dynamic, INT4 g64 и INT2 g64.

### Model-size quality scaling

```text
quality_hqq_qwen06.sbatch
quality_hqq_qwen17.sbatch
quality_hqq_qwen4b.sbatch
quality_hqq_qwen8b.sbatch
quality_hqq_qwen14b.sbatch
```

Qwen3-14B используется только для quality и запускается на двух V100.

### Qwen3-1.7B group-size ablation

```text
quality_hqq_ablation_qwen17.sbatch
final_hqq_ablation_qwen17.sbatch
```

Первый job дал ранние ablation-точки (`INT8 g64`, `INT4 g16/g128`).

`final_hqq_ablation_qwen17.sbatch` закрыл недостающие quality cells и запустил финальную performance matrix:

```text
Dynamic
INT8 g64
INT4 g16/g32/g64/g128
INT2 g16/g32/g64/g128
```

### Short-context control

```text
final_short_quality_qwen06.sbatch
```

Проверяет Qwen3-0.6B на 2048/4096 токенах для Dynamic, INT4 g64 и INT2 g64.

## Baseline и вторичные jobs

```text
pilot_qwen06.sbatch
pilot_prefill_decode_qwen06.sbatch
full_dynamic_static_qwen06.sbatch
long_context_dynamic_static_qwen06.sbatch
bench_static_compile_qwen06.sbatch
```

Они относятся к раннему FP16 baseline и StaticCache/compile части исследования.

`bench_qwen06.sbatch` и другие ранние `bench_*` jobs сохранены как история настройки benchmark pipeline.

## Correctness, smoke и diagnostics

```text
check_cache_correctness_qwen06.sbatch
check_quantized_cache_qwen06.sbatch
check_quantized_cache_hqq_qwen06.sbatch
check_static_compile_qwen06.sbatch
diagnose_hqq_int3.sbatch
diagnose_offloaded_qwen06.sbatch
diagnose_offloaded_qwen06_main.sbatch
diagnose_offloaded_a100.sbatch
quality_hqq_qwen14b_smoke.sbatch
smoke_hqq_qwen8b_40k.sbatch
```

Эти jobs использовались для проверки инфраструктуры, compatibility и correctness. Они не являются основной experimental matrix.

A100/H100/H200 в итоге не использовались в основных результатах; доступ аккаунта к этим типам узлов был ограничен.

## Дополнительные ablation jobs

```text
perf_hqq_ablation_qwen17.sbatch
```

Промежуточный performance ablation. Финальная Qwen3-1.7B performance matrix была затем выполнена через `final_hqq_ablation_qwen17.sbatch`.

## Как читать результаты

Slurm stdout/stderr пишутся в локальную папку:

```text
logs/
```

Она исключена из Git.

Raw CSV сохраняются в:

```text
results/raw/
```

Финальный анализ:

```bash
python src/analyze_final.py
```

Canonical summary:

```text
results/summary/final/
```

Canonical plots:

```text
plots/final/
```

## Важно при повторном запуске

Имена raw CSV содержат `$SLURM_JOB_ID`. Committed canonical CSV соответствуют конкретным запускам, на которых построен финальный отчёт.

Новый `sbatch` создаст файл с новым job ID. Чтобы использовать новый run как canonical input, нужно явно заменить соответствующий input в `src/analyze_final.py` или сохранить его под ожидаемым canonical именем после проверки результата.
