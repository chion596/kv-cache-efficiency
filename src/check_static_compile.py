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
    CompileConfig,
)


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
def run_generate(
    model,
    inputs,
    cache_impl,
    new_tokens,
    compile_enabled,
    compile_config=None,
):
    kwargs = dict(
        **inputs,
        do_sample=False,
        use_cache=True,
        cache_implementation=cache_impl,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        pad_token_id=model.generation_config.pad_token_id,
        disable_compile=not compile_enabled,
    )

    if compile_enabled:
        kwargs["compile_config"] = compile_config

    torch.cuda.synchronize()
    start = time.perf_counter()

    output = model.generate(**kwargs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    context_len = inputs["input_ids"].shape[-1]
    generated = output[0, context_len:].cpu()

    return {
        "elapsed": elapsed,
        "hash": token_hash(generated),
        "tokens": generated,
    }


def print_dynamo_counters():
    try:
        counters = torch._dynamo.utils.counters

        print()
        print("Torch Dynamo counters:")

        for name in [
            "frames",
            "stats",
            "inductor",
            "graph_break",
        ]:
            if name in counters:
                print(
                    f"{name}:",
                    dict(counters[name]),
                )

    except Exception as exc:
        print(
            "Could not read Dynamo counters:",
            repr(exc),
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(42)

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

    compile_config = CompileConfig()

    print("=" * 72)
    print("STATIC CACHE COMPILE CHECK")
    print("=" * 72)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("Transformers:", transformers.__version__)
    print("Context:", args.context)
    print("New tokens:", args.new_tokens)
    print("Warmups:", args.warmups)
    print("Measured repeats:", args.repeats)
    print(
        "Compile config:",
        compile_config.to_dict(),
    )
    print("=" * 72)

    # ----------------------------------------------------------
    # Dynamic correctness reference
    # ----------------------------------------------------------

    print()
    print("===== DYNAMIC REFERENCE =====")

    dynamic = run_generate(
        model=model,
        inputs=inputs,
        cache_impl="dynamic",
        new_tokens=args.new_tokens,
        compile_enabled=False,
    )

    print(
        f"time={dynamic['elapsed']:.3f}s "
        f"hash={dynamic['hash']}"
    )

    # ----------------------------------------------------------
    # Static eager
    # ----------------------------------------------------------

    print()
    print("===== STATIC EAGER =====")

    eager_times = []
    eager_hashes = []

    # one unmeasured warmup
    warm = run_generate(
        model=model,
        inputs=inputs,
        cache_impl="static",
        new_tokens=args.new_tokens,
        compile_enabled=False,
    )

    print(
        "warmup:",
        f"{warm['elapsed']:.3f}s",
        f"hash={warm['hash']}",
        f"same_as_dynamic="
        f"{torch.equal(warm['tokens'], dynamic['tokens'])}",
    )

    for i in range(args.repeats):
        result = run_generate(
            model=model,
            inputs=inputs,
            cache_impl="static",
            new_tokens=args.new_tokens,
            compile_enabled=False,
        )

        eager_times.append(result["elapsed"])
        eager_hashes.append(result["hash"])

        print(
            f"repeat={i} "
            f"time={result['elapsed']:.3f}s "
            f"hash={result['hash']} "
            f"same_as_dynamic="
            f"{torch.equal(result['tokens'], dynamic['tokens'])}"
        )

    # ----------------------------------------------------------
    # Reset compiler state before compile experiment
    # ----------------------------------------------------------

    try:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
    except Exception:
        pass

    # ----------------------------------------------------------
    # Static compiled warmup
    # ----------------------------------------------------------

    print()
    print("===== STATIC COMPILED WARMUP =====")

    compile_warmup_times = []

    for i in range(args.warmups):
        result = run_generate(
            model=model,
            inputs=inputs,
            cache_impl="static",
            new_tokens=args.new_tokens,
            compile_enabled=True,
            compile_config=compile_config,
        )

        compile_warmup_times.append(
            result["elapsed"]
        )

        print(
            f"warmup={i} "
            f"time={result['elapsed']:.3f}s "
            f"hash={result['hash']} "
            f"same_as_dynamic="
            f"{torch.equal(result['tokens'], dynamic['tokens'])}"
        )

    print(
        "compiled_call_exists:",
        hasattr(model, "_compiled_call"),
    )

    # ----------------------------------------------------------
    # Static compiled measurements
    # ----------------------------------------------------------

    print()
    print("===== STATIC COMPILED MEASURED =====")

    compiled_times = []
    compiled_hashes = []

    for i in range(args.repeats):
        result = run_generate(
            model=model,
            inputs=inputs,
            cache_impl="static",
            new_tokens=args.new_tokens,
            compile_enabled=True,
            compile_config=compile_config,
        )

        compiled_times.append(result["elapsed"])
        compiled_hashes.append(result["hash"])

        print(
            f"repeat={i} "
            f"time={result['elapsed']:.3f}s "
            f"hash={result['hash']} "
            f"same_as_dynamic="
            f"{torch.equal(result['tokens'], dynamic['tokens'])}"
        )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    eager_mean = statistics.mean(eager_times)
    compiled_mean = statistics.mean(compiled_times)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        "static eager mean:",
        f"{eager_mean:.4f}s",
    )

    print(
        "static compiled mean:",
        f"{compiled_mean:.4f}s",
    )

    print(
        "compiled/eager ratio:",
        f"{compiled_mean / eager_mean:.4f}",
    )

    print(
        "speedup:",
        f"{eager_mean / compiled_mean:.4f}x",
    )

    print(
        "eager unique hashes:",
        len(set(eager_hashes)),
    )

    print(
        "compiled unique hashes:",
        len(set(compiled_hashes)),
    )

    print(
        "compile warmup times:",
        ", ".join(
            f"{x:.3f}s"
            for x in compile_warmup_times
        ),
    )

    print_dynamo_counters()

    print("=" * 72)


if __name__ == "__main__":
    main()
