# StaticCache и torch.compile

## Конфигурация

- модель: Qwen3-0.6B;
- GPU: NVIDIA Tesla V100-SXM2 32 GB;
- PyTorch: 2.14.0+cu126;
- Transformers: 5.17.0;
- dtype: FP16;
- output: 64 токена;
- 5 измерений после warm-up;
- compile backend: inductor;
- compile mode: reduce-overhead.

Сравнивались:

- DynamicCache без compilation;
- StaticCache без compilation;
- StaticCache со штатной automatic compilation в `generate()`.

Prefill при этом остаётся eager, а compiled forward используется
для итеративного decode.

## Корректность

Во всех исследованных конфигурациях:

- каждый режим воспроизводим между повторами;
- Static eager совпадает с DynamicCache по output;
- Static compiled совпадает с DynamicCache по output.

Graph breaks не наблюдались.

## Compilation overhead

При первом использовании compiled StaticCache:

- context=2048: первый warm-up занял 89.98 s;
- был построен первый compiled graph.

При context=8192:

- первый warm-up занял 52.09 s;
- появился второй compiled graph.

После этого число compiled graphs осталось равным двум
вплоть до context=40895.

Таким образом, cold compilation имеет большой разовый overhead,
который необходимо исключать из steady-state performance.

## Steady-state latency

Средняя end-to-end latency:

| Context | Dynamic eager | Static eager | Static compiled |
|---:|---:|---:|---:|
| 2048 | 2.149 s | 2.472 s | 0.829 s |
| 8192 | 2.486 s | 2.828 s | 2.213 s |
| 12288 | 3.080 s | 3.645 s | 3.310 s |
| 16384 | 4.159 s | 4.934 s | 4.495 s |
| 32768 | 9.795 s | 11.237 s | 10.623 s |
| 40895 | 13.443 s | 15.229 s | 14.523 s |

## Эффект compilation

Относительно StaticCache без compilation снижение latency составляет:

- 2048: 66.5%;
- 8192: 21.7%;
- 12288: 9.2%;
- 16384: 8.9%;
- 32768: 5.5%;
- 40895: 4.6%.

Таким образом, выигрыш compilation уменьшается с увеличением
длины входного контекста.

Это согласуется с тем, что automatic compilation используется
преимущественно для decode, тогда как стоимость eager prefill
с ростом длины контекста становится всё существеннее.

## Crossover с DynamicCache

На context=2048 и context=8192 StaticCache с compilation
быстрее DynamicCache по end-to-end latency.

При context=12288 Static compiled уже примерно на 7.5%
медленнее DynamicCache.

На более длинных контекстах разница остаётся примерно на уровне
8% в пользу DynamicCache.

Следовательно, в исследованной конфигурации практический crossover
между compiled StaticCache и DynamicCache находится между
8192 и 12288 токенами.

Точный порог по текущей дискретной сетке определять нельзя.

## Память

Static compiled и Static eager имеют практически одинаковый
peak allocated GPU memory.

Следовательно, compilation изменяет вычислительную
производительность, но не даёт заметной дополнительной экономии
памяти KV Cache.

## Вывод

StaticCache без compilation уступает DynamicCache по скорости.

Использование torch.compile существенно меняет результат на коротких
и средних контекстах, однако преимущество уменьшается с ростом
контекста.

На длинных контекстах DynamicCache снова становится быстрее
StaticCache даже после compilation.

Кроме steady-state latency необходимо учитывать большой разовый
cost compilation.
