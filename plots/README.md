# `plots/`: графики экспериментов

## Финальные графики

Графики, используемые в корневом `README.md`:

```text
plots/final/
```

Сейчас это:

```text
memory_vs_context_qwen17.png
decode_speed_vs_context_qwen17.png
model_size_quality.png
int4_group_size_quality.png
quality_memory_tradeoff.png
```

Они строятся через:

```bash
python src/analyze_final.py
```

## Остальные директории

- `baseline/` — ранний DynamicCache/StaticCache baseline;
- `long_context/` — FP16 long-context sweep;
- `quality_hqq/` — ранний HQQ quality experiment;
- `quantized_scaling/` — промежуточные HQQ scaling plots;
- `static_compile/` — StaticCache + `torch.compile`.

Эти графики сохраняются как история исследования, но не используются как основной источник итоговых выводов.
