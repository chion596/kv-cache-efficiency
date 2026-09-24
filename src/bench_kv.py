import argparse
import csv
import gc
import math
import os
import platform
import time
import traceback

import psutil
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_input(tokenizer, context_len, device):
    """
    Create exactly context_len valid tokens.
    Same underlying text distribution for every experiment.
    """
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


@torch.inference_mode()
def run_generation(
    model,
    inputs,
    cache_impl,
    new_tokens,
):
    return model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        cache_implementation=cache_impl,

        # Force approximately identical generation work.
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,

        # Important: isolate cache strategy itself.
        # static+compile will be a separate experiment.
        disable_compile=True,

        pad_token_id=model.generation_config.pad_token_id,
    )


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
        default=["dynamic", "static", "offloaded"],
    )

    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    torch.manual_seed(42)

    device = "cuda"

    print("=" * 70)
    print("KV CACHE BENCHMARK")
    print("=" * 70)
    print("Model:", args.model)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("Transformers:", transformers.__version__)
    print("Python:", platform.python_version())
    print("Contexts:", args.contexts)
    print("Caches:", args.caches)
    print("Repeats:", args.repeats)
    print("New tokens:", args.new_tokens)
    print("=" * 70)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        local_files_only=True,
    ).to(device)

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    torch.cuda.synchronize()

    model_memory_gb = (
        torch.cuda.memory_allocated() / 2**30
    )

    print(
        "Model GPU memory after loading:",
        round(model_memory_gb, 3),
        "GB",
    )

    fields = [
        "model",
        "gpu",
        "torch_version",
        "transformers_version",
        "context_len",
        "cache",
        "repeat",
        "generated_tokens",
        "elapsed_sec",
        "e2e_tokens_per_sec",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "host_rss_gb",
        "error",
    ]

    out_dir = os.path.dirname(args.out)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        f.flush()

        for cache_impl in args.caches:
            for context_len in args.contexts:

                print()
                print("-" * 70)
                print(
                    f"CACHE={cache_impl} "
                    f"CONTEXT={context_len}"
                )
                print("-" * 70)

                inputs = make_input(
                    tokenizer,
                    context_len,
                    device,
                )

                # Warm-up: do not record this run.
                print("Warm-up...")

                try:
                    _ = run_generation(
                        model=model,
                        inputs=inputs,
                        cache_impl=cache_impl,
                        new_tokens=8,
                    )

                    torch.cuda.synchronize()

                except Exception as exc:
                    print(
                        "WARM-UP FAILED:",
                        repr(exc),
                    )

                    writer.writerow({
                        "model": args.model,
                        "gpu": torch.cuda.get_device_name(0),
                        "torch_version": torch.__version__,
                        "transformers_version": transformers.__version__,
                        "context_len": context_len,
                        "cache": cache_impl,
                        "repeat": -1,
                        "generated_tokens": "",
                        "elapsed_sec": "",
                        "e2e_tokens_per_sec": "",
                        "peak_allocated_gb": "",
                        "peak_reserved_gb": "",
                        "host_rss_gb": "",
                        "error": f"warmup: {repr(exc)}",
                    })

                    f.flush()

                    del inputs
                    gc.collect()
                    torch.cuda.empty_cache()

                    continue

                for repeat in range(args.repeats):

                    print(
                        f"Run {repeat + 1}/{args.repeats}"
                    )

                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                    try:
                        start = time.perf_counter()

                        output = run_generation(
                            model=model,
                            inputs=inputs,
                            cache_impl=cache_impl,
                            new_tokens=args.new_tokens,
                        )

                        torch.cuda.synchronize()

                        elapsed = (
                            time.perf_counter() - start
                        )

                        generated_tokens = (
                            output.shape[-1]
                            - inputs["input_ids"].shape[-1]
                        )

                        peak_allocated = (
                            torch.cuda.max_memory_allocated()
                            / 2**30
                        )

                        peak_reserved = (
                            torch.cuda.max_memory_reserved()
                            / 2**30
                        )

                        host_rss = (
                            psutil.Process()
                            .memory_info()
                            .rss
                            / 2**30
                        )

                        throughput = (
                            generated_tokens / elapsed
                        )

                        row = {
                            "model": args.model,
                            "gpu": torch.cuda.get_device_name(0),
                            "torch_version": torch.__version__,
                            "transformers_version": transformers.__version__,
                            "context_len": context_len,
                            "cache": cache_impl,
                            "repeat": repeat,
                            "generated_tokens": generated_tokens,
                            "elapsed_sec": elapsed,
                            "e2e_tokens_per_sec": throughput,
                            "peak_allocated_gb": peak_allocated,
                            "peak_reserved_gb": peak_reserved,
                            "host_rss_gb": host_rss,
                            "error": "",
                        }

                        writer.writerow(row)
                        f.flush()

                        print(
                            f"time={elapsed:.3f}s "
                            f"tok/s={throughput:.3f} "
                            f"peak={peak_allocated:.3f}GB "
                            f"reserved={peak_reserved:.3f}GB"
                        )

                        del output

                    except Exception as exc:
                        traceback.print_exc()

                        writer.writerow({
                            "model": args.model,
                            "gpu": torch.cuda.get_device_name(0),
                            "torch_version": torch.__version__,
                            "transformers_version": transformers.__version__,
                            "context_len": context_len,
                            "cache": cache_impl,
                            "repeat": repeat,
                            "generated_tokens": "",
                            "elapsed_sec": "",
                            "e2e_tokens_per_sec": "",
                            "peak_allocated_gb": "",
                            "peak_reserved_gb": "",
                            "host_rss_gb": "",
                            "error": repr(exc),
                        })

                        f.flush()

                        torch.cuda.empty_cache()

                del inputs
                gc.collect()
                torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("FINISHED")
    print("Results:", args.out)
    print("=" * 70)


if __name__ == "__main__":
    main()
