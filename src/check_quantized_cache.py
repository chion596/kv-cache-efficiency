import argparse
import hashlib
import math
import statistics
import time

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


GIB = 2 ** 30


def make_input(tokenizer, context_len, device):
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

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def token_hash(ids):
    payload = ",".join(
        str(x) for x in ids.tolist()
    ).encode()

    return hashlib.sha256(payload).hexdigest()[:16]


@torch.inference_mode()
def run(
    model,
    inputs,
    mode,
    new_tokens,
):
    if mode == "dynamic":
        cache_implementation = "dynamic"
        cache_config = None

    elif mode == "int4":
        cache_implementation = "quantized"
        cache_config = {
            "backend": "quanto",
            "nbits": 4,
            "axis_key": 0,
            "axis_value": 0,
            "q_group_size": 64,
            "residual_length": 128,
        }

    elif mode == "int2":
        cache_implementation = "quantized"
        cache_config = {
            "backend": "quanto",
            "nbits": 2,
            "axis_key": 0,
            "axis_value": 0,
            "q_group_size": 64,
            "residual_length": 128,
        }

    else:
        raise ValueError(mode)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    kwargs = dict(
        **inputs,
        do_sample=False,
        use_cache=True,
        cache_implementation=cache_implementation,
        disable_compile=True,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        pad_token_id=model.generation_config.pad_token_id,
    )

    if cache_config is not None:
        kwargs["cache_config"] = cache_config

    torch.cuda.synchronize()
    start = time.perf_counter()

    output = model.generate(**kwargs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak = (
        torch.cuda.max_memory_allocated()
        / GIB
    )

    context_len = inputs["input_ids"].shape[-1]

    generated = (
        output[0, context_len:]
        .detach()
        .cpu()
    )

    return {
        "elapsed": elapsed,
        "peak": peak,
        "hash": token_hash(generated),
        "tokens": generated,
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)

    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )

    inputs = make_input(
        tokenizer,
        args.context,
        "cuda",
    )

    print("=" * 72)
    print("QUANTIZED KV CACHE SMOKE TEST")
    print("=" * 72)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("Context:", args.context)
    print("New tokens:", args.new_tokens)
    print("=" * 72)

    # Global warmup
    print("Global dynamic warmup...")

    _ = run(
        model,
        inputs,
        "dynamic",
        4,
    )

    results = {}

    for mode in ["dynamic", "int4", "int2"]:
        print()
        print("=" * 72)
        print("MODE:", mode)
        print("=" * 72)

        # same branch warmup
        warm = run(
            model,
            inputs,
            mode,
            args.new_tokens,
        )

        print(
            f"warmup "
            f"time={warm['elapsed']:.3f}s "
            f"peak={warm['peak']:.3f}GiB "
            f"hash={warm['hash']}"
        )

        times = []
        peaks = []
        hashes = []
        tokens = []

        for r in range(args.repeats):
            result = run(
                model,
                inputs,
                mode,
                args.new_tokens,
            )

            times.append(result["elapsed"])
            peaks.append(result["peak"])
            hashes.append(result["hash"])
            tokens.append(result["tokens"])

            print(
                f"repeat={r} "
                f"time={result['elapsed']:.3f}s "
                f"peak={result['peak']:.3f}GiB "
                f"hash={result['hash']}"
            )

        stable = all(
            torch.equal(tokens[0], x)
            for x in tokens[1:]
        )

        results[mode] = tokens[0]

        print(
            f"mean_time={statistics.mean(times):.3f}s "
            f"mean_peak={statistics.mean(peaks):.3f}GiB "
            f"unique_hashes={len(set(hashes))} "
            f"self_consistent={stable}"
        )

    print()
    print("=" * 72)
    print("OUTPUT COMPARISON")
    print("=" * 72)

    for mode in ["int4", "int2"]:
        same = torch.equal(
            results["dynamic"],
            results[mode],
        )

        print(
            f"dynamic == {mode}: {same}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
