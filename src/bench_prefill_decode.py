import argparse
import csv
import gc
import hashlib
import math
import os
import platform
import time
import traceback

import psutil
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    StaticCache,
)


GIB = 2 ** 30


def make_input(tokenizer, context_len, device):
    """Создаёт вход ровно из context_len токенов."""
    text = (
        "Transformer inference uses a key value cache to reuse "
        "attention states from previous tokens. "
    )

    base = tokenizer(
        text,
        add_special_tokens=False,
    ).input_ids

    repeats = math.ceil(context_len / len(base))
    ids = (base * repeats)[:context_len]

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def make_cache(model, cache_impl, max_cache_len):
    """Создаёт KV cache нужного типа."""
    if cache_impl == "dynamic":
        return DynamicCache(
            config=model.config,
        )

    if cache_impl == "static":
        return StaticCache(
            config=model.config,
            max_cache_len=max_cache_len,
        )

    if cache_impl == "offloaded":
        return DynamicCache(
            config=model.config,
            offloading=True,
        )

    raise ValueError(
        f"Unknown cache implementation: {cache_impl}"
    )


def tensor_bytes(tensor):
    if tensor is None:
        return 0

    return tensor.numel() * tensor.element_size()


def cache_storage(cache):
    """
    Считает физически выделенную память под tensors KV cache
    отдельно на GPU и CPU.
    """
    gpu_bytes = 0
    cpu_bytes = 0
    other_bytes = 0

    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)

            if not isinstance(tensor, torch.Tensor):
                continue

            size = tensor_bytes(tensor)

            if tensor.device.type == "cuda":
                gpu_bytes += size
            elif tensor.device.type == "cpu":
                cpu_bytes += size
            else:
                other_bytes += size

    return {
        "gpu": gpu_bytes,
        "cpu": cpu_bytes,
        "other": other_bytes,
        "total": gpu_bytes + cpu_bytes + other_bytes,
    }


def theoretical_kv_bytes(config, seq_len, dtype):
    """
    KV bytes =
        2 (K и V)
        * layers
        * KV heads
        * head_dim
        * sequence length
        * bytes per element
    """
    layers = config.num_hidden_layers

    kv_heads = getattr(
        config,
        "num_key_value_heads",
        config.num_attention_heads,
    )

    head_dim = getattr(
        config,
        "head_dim",
        config.hidden_size // config.num_attention_heads,
    )

    dtype_bytes = torch.empty(
        (),
        dtype=dtype,
    ).element_size()

    return (
        2
        * layers
        * kv_heads
        * head_dim
        * seq_len
        * dtype_bytes
    )


def token_hash(generated_tokens):
    """Короткий checksum output для проверки идентичности cache strategies."""
    ids = torch.cat(
        generated_tokens,
        dim=-1,
    ).detach().cpu().flatten().tolist()

    payload = ",".join(
        str(x) for x in ids
    ).encode()

    return hashlib.sha256(payload).hexdigest()[:16]


@torch.inference_mode()
def measure_once(
    model,
    inputs,
    cache_impl,
    new_tokens,
    max_cache_len,
    model_memory_gib,
):
    context_len = inputs["input_ids"].shape[-1]
    device = inputs["input_ids"].device
    dtype = next(model.parameters()).dtype

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cache = make_cache(
        model,
        cache_impl,
        max_cache_len=max_cache_len,
    )

    attention_mask = inputs["attention_mask"].clone()

    cache_position = torch.arange(
        context_len,
        dtype=torch.int64,
        device=device,
    )

    # ============================================================
    # PREFILL
    # ============================================================

    torch.cuda.synchronize()
    request_start = time.perf_counter()

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=cache,
        use_cache=True,

        # Нам нужен только logits последней позиции.
        # Иначе при длинном контексте создаётся огромный tensor
        # [batch, context, vocab].
        logits_to_keep=1,
    )

    torch.cuda.synchronize()

    prefill_end = time.perf_counter()
    prefill_sec = prefill_end - request_start

    cache = outputs.past_key_values

    # Первый output-token уже определяется после prefill.
    first_token = outputs.logits[:, -1:].argmax(
        dim=-1
    )

    torch.cuda.synchronize()

    first_token_end = time.perf_counter()
    ttft_sec = first_token_end - request_start

    generated = [first_token]

    prefill_peak_allocated = (
        torch.cuda.max_memory_allocated()
        / GIB
    )

    prefill_peak_reserved = (
        torch.cuda.max_memory_reserved()
        / GIB
    )

    prompt_cache_storage = cache_storage(cache)

    del outputs

    # ============================================================
    # DECODE
    # ============================================================

    decode_steps = max(new_tokens - 1, 0)

    decode_sec = 0.0
    decode_peak_allocated = (
        torch.cuda.memory_allocated() / GIB
    )
    decode_peak_reserved = (
        torch.cuda.memory_reserved() / GIB
    )

    if decode_steps > 0:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        decode_start = time.perf_counter()

        next_token = first_token

        # Следующая позиция после последнего prompt-token.
        current_cache_position = (
            cache_position[-1:] + 1
        )

        for _ in range(decode_steps):
            attention_mask = torch.cat(
                [
                    attention_mask,
                    attention_mask.new_ones(
                        (attention_mask.shape[0], 1)
                    ),
                ],
                dim=-1,
            )

            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                cache_position=current_cache_position,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )

            cache = outputs.past_key_values

            next_token = outputs.logits[
                :, -1:
            ].argmax(dim=-1)

            generated.append(next_token)

            current_cache_position = (
                current_cache_position + 1
            )

            del outputs

        torch.cuda.synchronize()

        decode_sec = (
            time.perf_counter()
            - decode_start
        )

        decode_peak_allocated = (
            torch.cuda.max_memory_allocated()
            / GIB
        )

        decode_peak_reserved = (
            torch.cuda.max_memory_reserved()
            / GIB
        )

    final_cache_storage = cache_storage(cache)

    cache_seq_len = int(
        cache.get_seq_length()
    )

    total_peak_allocated = max(
        prefill_peak_allocated,
        decode_peak_allocated,
    )

    total_peak_reserved = max(
        prefill_peak_reserved,
        decode_peak_reserved,
    )

    e2e_sec = ttft_sec + decode_sec

    prefill_tok_s = (
        context_len / prefill_sec
    )

    decode_tok_s = (
        decode_steps / decode_sec
        if decode_sec > 0
        else float("nan")
    )

    tpot_ms = (
        1000.0 * decode_sec / decode_steps
        if decode_steps > 0
        else float("nan")
    )

    e2e_tok_s = (
        new_tokens / e2e_sec
        if e2e_sec > 0
        else float("nan")
    )

    theoretical_prompt = theoretical_kv_bytes(
        model.config,
        context_len,
        dtype,
    )

    theoretical_final = theoretical_kv_bytes(
        model.config,
        cache_seq_len,
        dtype,
    )

    host_rss_gib = (
        psutil.Process()
        .memory_info()
        .rss
        / GIB
    )

    return {
        "generated_tokens": new_tokens,
        "decode_steps": decode_steps,

        "prefill_sec": prefill_sec,
        "prefill_tok_s": prefill_tok_s,

        "ttft_ms": ttft_sec * 1000.0,

        "decode_sec": decode_sec,
        "decode_tok_s": decode_tok_s,
        "tpot_ms": tpot_ms,

        "e2e_sec": e2e_sec,
        "e2e_tok_s": e2e_tok_s,

        "model_memory_gib": model_memory_gib,

        "prefill_peak_allocated_gib":
            prefill_peak_allocated,
        "decode_peak_allocated_gib":
            decode_peak_allocated,
        "total_peak_allocated_gib":
            total_peak_allocated,

        "prefill_peak_reserved_gib":
            prefill_peak_reserved,
        "decode_peak_reserved_gib":
            decode_peak_reserved,
        "total_peak_reserved_gib":
            total_peak_reserved,

        "peak_extra_over_model_gib":
            total_peak_allocated
            - model_memory_gib,

        "cache_seq_len": cache_seq_len,

        "theoretical_kv_prompt_gib":
            theoretical_prompt / GIB,
        "theoretical_kv_final_gib":
            theoretical_final / GIB,

        "cache_prompt_gpu_gib":
            prompt_cache_storage["gpu"] / GIB,
        "cache_prompt_cpu_gib":
            prompt_cache_storage["cpu"] / GIB,
        "cache_prompt_total_gib":
            prompt_cache_storage["total"] / GIB,

        "cache_final_gpu_gib":
            final_cache_storage["gpu"] / GIB,
        "cache_final_cpu_gib":
            final_cache_storage["cpu"] / GIB,
        "cache_final_total_gib":
            final_cache_storage["total"] / GIB,

        "host_rss_gib": host_rss_gib,

        "output_hash": token_hash(generated),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)

    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[512, 2048, 4096, 8192, 16384],
    )

    parser.add_argument(
        "--caches",
        nargs="+",
        default=[
            "dynamic",
            "static",
            "offloaded",
        ],
    )

    parser.add_argument(
        "--new-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available"
        )

    torch.manual_seed(42)

    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        local_files_only=True,
    ).to(device)

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    torch.cuda.synchronize()

    model_memory_gib = (
        torch.cuda.memory_allocated()
        / GIB
    )

    print("=" * 72)
    print("PREFILL / DECODE KV CACHE BENCHMARK")
    print("=" * 72)
    print("Model:", args.model)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("Transformers:", transformers.__version__)
    print("Python:", platform.python_version())
    print("Model memory:", f"{model_memory_gib:.3f} GiB")
    print("Contexts:", args.contexts)
    print("Caches:", args.caches)
    print("Repeats:", args.repeats)
    print("New tokens:", args.new_tokens)

    cfg = model.config

    print(
        "Architecture:",
        f"layers={cfg.num_hidden_layers},",
        f"attention_heads={cfg.num_attention_heads},",
        f"kv_heads={cfg.num_key_value_heads},",
        f"head_dim={cfg.head_dim}",
    )

    kv_per_token = theoretical_kv_bytes(
        cfg,
        1,
        next(model.parameters()).dtype,
    )

    print(
        "Theoretical KV per token:",
        f"{kv_per_token / 1024:.1f} KiB",
    )
    print("=" * 72)

    fields = [
        "model",
        "gpu",
        "torch_version",
        "transformers_version",
        "context_len",
        "cache",
        "repeat",
        "generated_tokens",
        "decode_steps",

        "prefill_sec",
        "prefill_tok_s",
        "ttft_ms",

        "decode_sec",
        "decode_tok_s",
        "tpot_ms",

        "e2e_sec",
        "e2e_tok_s",

        "model_memory_gib",

        "prefill_peak_allocated_gib",
        "decode_peak_allocated_gib",
        "total_peak_allocated_gib",

        "prefill_peak_reserved_gib",
        "decode_peak_reserved_gib",
        "total_peak_reserved_gib",

        "peak_extra_over_model_gib",

        "cache_seq_len",

        "theoretical_kv_prompt_gib",
        "theoretical_kv_final_gib",

        "cache_prompt_gpu_gib",
        "cache_prompt_cpu_gib",
        "cache_prompt_total_gib",

        "cache_final_gpu_gib",
        "cache_final_cpu_gib",
        "cache_final_total_gib",

        "host_rss_gib",
        "output_hash",
        "error",
    ]

    out_dir = os.path.dirname(args.out)

    if out_dir:
        os.makedirs(
            out_dir,
            exist_ok=True,
        )

    with open(
        args.out,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        f.flush()

        for cache_impl in args.caches:
            for context_len in args.contexts:

                print()
                print("-" * 72)
                print(
                    f"CACHE={cache_impl} "
                    f"CONTEXT={context_len}"
                )
                print("-" * 72)

                inputs = make_input(
                    tokenizer,
                    context_len,
                    device,
                )

                max_cache_len = (
                    context_len
                    + args.new_tokens
                    + 1
                )

                # Прогрев той же кодовой ветки.
                print("Warm-up...")

                try:
                    _ = measure_once(
                        model=model,
                        inputs=inputs,
                        cache_impl=cache_impl,
                        new_tokens=min(
                            4,
                            args.new_tokens,
                        ),
                        max_cache_len=max_cache_len,
                        model_memory_gib=model_memory_gib,
                    )

                except Exception as exc:
                    traceback.print_exc()

                    writer.writerow({
                        "model": args.model,
                        "gpu":
                            torch.cuda.get_device_name(0),
                        "torch_version":
                            torch.__version__,
                        "transformers_version":
                            transformers.__version__,
                        "context_len":
                            context_len,
                        "cache":
                            cache_impl,
                        "repeat":
                            -1,
                        "error":
                            f"warmup: {repr(exc)}",
                    })

                    f.flush()

                    del inputs
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                for repeat in range(
                    args.repeats
                ):
                    print(
                        f"Run "
                        f"{repeat + 1}/"
                        f"{args.repeats}"
                    )

                    try:
                        result = measure_once(
                            model=model,
                            inputs=inputs,
                            cache_impl=cache_impl,
                            new_tokens=args.new_tokens,
                            max_cache_len=max_cache_len,
                            model_memory_gib=model_memory_gib,
                        )

                        row = {
                            "model": args.model,
                            "gpu":
                                torch.cuda.get_device_name(0),
                            "torch_version":
                                torch.__version__,
                            "transformers_version":
                                transformers.__version__,
                            "context_len":
                                context_len,
                            "cache":
                                cache_impl,
                            "repeat":
                                repeat,
                            **result,
                            "error": "",
                        }

                        writer.writerow(row)
                        f.flush()

                        print(
                            f"prefill="
                            f"{result['prefill_sec']:.3f}s "
                            f"TTFT="
                            f"{result['ttft_ms']:.1f}ms "
                            f"decode="
                            f"{result['decode_tok_s']:.2f} tok/s "
                            f"TPOT="
                            f"{result['tpot_ms']:.2f}ms "
                            f"peak="
                            f"{result['total_peak_allocated_gib']:.3f}GiB "
                            f"hash="
                            f"{result['output_hash']}"
                        )

                    except Exception as exc:
                        traceback.print_exc()

                        writer.writerow({
                            "model":
                                args.model,
                            "gpu":
                                torch.cuda.get_device_name(0),
                            "torch_version":
                                torch.__version__,
                            "transformers_version":
                                transformers.__version__,
                            "context_len":
                                context_len,
                            "cache":
                                cache_impl,
                            "repeat":
                                repeat,
                            "error":
                                repr(exc),
                        })

                        f.flush()

                del inputs
                gc.collect()
                torch.cuda.empty_cache()

    print()
    print("=" * 72)
    print("FINISHED")
    print("Results:", args.out)
    print("=" * 72)


if __name__ == "__main__":
    main()
