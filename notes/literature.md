# Литература и связь с работой

## Hugging Face QuantizedCache

Основные эксперименты используют `QuantizedCache` из Transformers 5.17.0 с HQQ backend. Для описания cache API и рекомендуемых параметров используется документация именно этой версии: [Transformers 5.17.0 — Cache strategies](https://huggingface.co/docs/transformers/v5.17.0/kv_cache).

```text
backend = hqq
axis_key = 1
axis_value = 1
residual_length = 128
```

Основное сравнение моделей проводится при `q_group_size=64`. На Qwen3-1.7B дополнительно проверяются `g16`, `g32`, `g64`, `g128`.

Результаты относятся к этой реализации и конфигурации, а не к KV-cache quantization в целом.

### Почему quantized cache может быть медленнее

В протестированном `QuantizedLayer` хранятся:

- quantized K/V;
- residual K/V в исходной точности.

При update квантованная часть деквантуется перед использованием в attention. Поэтому уменьшение storage не обязано уменьшать latency. Это согласуется с экспериментом на V100: INT4/INT2 экономят память, но замедляют decode.

Это объяснение относится к конкретному Transformers/HQQ code path.

## KIVI

[KIVI](https://arxiv.org/abs/2402.02750) предлагает tuning-free low-bit KV-cache quantization и использует разные схемы для Keys и Values:

- Keys — per-channel;
- Values — per-token.

Мотивация авторов — различия распределений K/V и channel-wise outliers в Key cache.

Текущая работа **не воспроизводит KIVI**: здесь исследуется stock HQQ QuantizedCache с `axis_key=1`, `axis_value=1`.

### Связь с `q_group_size`

На Qwen3-1.7B при INT4:

```text
g16  -> 45/45 exact retrieval
g32  -> 37/45
g64  -> 14/45
g128 ->  4/45
```

Большая группа использует один набор параметров квантования для большего числа значений, поэтому рост quantization error является правдоподобным объяснением падения качества.

Однако текущие эксперименты не доказывают, что причиной являются именно Key outliers или механизм, описанный в KIVI. Для этого нужен отдельный ablation по схеме квантования K/V.

## Оценка storage при group quantization

Если на группу из `g` значений приходится:

- `b` бит на quantized value;
- FP16 `scale`;
- FP16 `zero`,

то приближённо:

```text
effective bits/value ~= b + 32/g
compression ~= 16 / (b + 32/g)
```

Для использованных конфигураций:

```text
INT4 g16 -> 2.67x
INT4 g64 -> 3.56x
INT2 g64 -> 6.40x
```

Эти оценки совпадают с измеренным физическим размером KV Cache.

Это не равно экономии всей GPU memory: веса модели и другие allocations остаются в памяти.

## Другие подходы

### H2O

[H2O](https://arxiv.org/abs/2306.14048) уменьшает KV Cache выбором части наиболее важных токенов. Это другой класс методов: уменьшается число хранимых состояний, а не точность каждого K/V.

### SnapKV

[SnapKV](https://arxiv.org/abs/2404.14469) также уменьшает число сохраняемых позиций на основе attention-паттернов. В текущих экспериментах H2O и SnapKV не реализуются и используются только как related work.

## Что получено в этой работе

Собственные экспериментальные результаты:

- измерение физического HQQ KV storage и total GPU memory;
- измерение decode slowdown на V100;
- сравнение устойчивости Qwen3 разных размеров;
- ablation по `q_group_size`;
- short-context control;
- диагностика Offloaded Cache, INT3 и Quanto.

Связь с литературой используется для интерпретации и постановки будущих экспериментов, а не как доказательство механизма наблюдаемых эффектов.

## Источники

1. Hugging Face. *Transformers v5.17.0 — Cache strategies*. [Documentation](https://huggingface.co/docs/transformers/v5.17.0/kv_cache).
2. Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*. arXiv:2402.02750, 2024. [arXiv](https://arxiv.org/abs/2402.02750).
3. Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. arXiv:2306.14048, 2023. [arXiv](https://arxiv.org/abs/2306.14048).
4. Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, Deming Chen. *SnapKV: LLM Knows What You are Looking for Before Generation*. arXiv:2404.14469, 2024. [arXiv](https://arxiv.org/abs/2404.14469).
