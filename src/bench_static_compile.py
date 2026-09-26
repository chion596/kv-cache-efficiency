import argparse
import csv
import gc
import hashlib
import math
import os
import platform
import socket
import time
import traceback

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CompileConfig,
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


def dynamo_stat(name, key):
    try:
        return int(
            torch._dynamo.utils.counters[name][key]
        )
    except Exception:
        return 0


@torch.inference_mode()
def run_once(
    model,
    inputs,
    mode,
    new_tokens,
    compile_config,
):
    if mode == "dynamic_eager":
        cache_impl = "dynamic"
        compile_enabled = False

    elif mode == "static_eager":
        cache_impl = "static"
        compile_enabled = False

    elif mode == "static_compiled":
        cache_impl = "static"
        compile_enabled = True

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

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

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / GIB
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / GIB
    )

    context_len = inputs["input_ids"].shape[-1]

    generated = (
        output[0, context_len:]
        .detach()
        .cpu()
    )

    generated_tokens = len(generated)

    result = {
        "elapsed_sec": elapsed,
        "generated_tokens": generated_tokens,
        "e2e_tok_s":
            generated_tokens / elapsed,

        "peak_allocated_gib":
            peak_allocated,
        "peak_reserved_gib":
            peak_reserved,

        "output_hash":
            token_hash(generated),

        "tokens":
            generated,

        "dynamo_unique_graphs":
            dynamo_stat(
                "stats",
                "unique_graphs",
            ),

        "dynamo_calls_captured":
            dynamo_stat(
                "stats",
                "calls_captured",
            ),
    }

    del output

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[
            2048,
            8192,
            12288,
            16384,
            32768,
            40895,
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
        default=5,
    )

    parser.add_argument(
        "--eager-warmups",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--compile-warmups",
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

    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )

    compile_config = CompileConfig()

    hostname = socket.gethostname()

    slurm_job_id = os.environ.get(
        "SLURM_JOB_ID",
        "unknown",
    )

    git_commit = os.environ.get(
        "GIT_COMMIT",
        "unknown",
    )

    print("=" * 76)
    print("STATIC CACHE COMPILE BENCHMARK")
    print("=" * 76)

    print("Model:", args.model)
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU capability:",
        torch.cuda.get_device_capability(0),
    )
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("Transformers:", transformers.__version__)
    print("Python:", platform.python_version())
    print("Hostname:", hostname)
    print("SLURM job:", slurm_job_id)
    print("Git commit:", git_commit)

    print("Contexts:", args.contexts)
    print("New tokens:", args.new_tokens)
    print("Repeats:", args.repeats)
    print(
        "Compile config:",
        compile_config.to_dict(),
    )

    print("=" * 76)

    # ----------------------------------------------------------
    # Global eager warmup.
    #
    # The smoke test showed that the very first model invocation
    # has a large cold-start overhead. We explicitly exclude it.
    # ----------------------------------------------------------

    print()
    print("GLOBAL MODEL WARMUP")

    global_inputs = make_input(
        tokenizer,
        512,
        device,
    )

    global_warmup = run_once(
        model=model,
        inputs=global_inputs,
        mode="dynamic_eager",
        new_tokens=8,
        compile_config=compile_config,
    )

    print(
        f"global warmup: "
        f"{global_warmup['elapsed_sec']:.3f}s"
    )

    del global_inputs
    gc.collect()
    torch.cuda.empty_cache()

    try:
        torch._dynamo.utils.counters.clear()
    except Exception:
        pass

    fields = [
        "model",
        "gpu",
        "gpu_capability",
        "torch_version",
        "torch_cuda",
        "transformers_version",
        "python_version",
        "hostname",
        "slurm_job_id",
        "git_commit",

        "context_len",
        "mode",
        "phase",
        "repeat",

        "generated_tokens",
        "elapsed_sec",
        "e2e_tok_s",

        "peak_allocated_gib",
        "peak_reserved_gib",

        "output_hash",
        "same_as_dynamic",

        "dynamo_unique_graphs",
        "dynamo_calls_captured",

        "error",
    ]

    out_dir = os.path.dirname(args.out)

    if out_dir:
        os.makedirs(
            out_dir,
            exist_ok=True,
        )

    modes = [
        "dynamic_eager",
        "static_eager",
        "static_compiled",
    ]

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

        for context_len in args.contexts:
            print()
            print("=" * 76)
            print(
                f"CONTEXT={context_len}"
            )
            print("=" * 76)

            inputs = make_input(
                tokenizer,
                context_len,
                device,
            )

            dynamic_reference = None

            for mode in modes:
                print()
                print("-" * 76)
                print(
                    f"MODE={mode} "
                    f"CONTEXT={context_len}"
                )
                print("-" * 76)

                if mode == "static_compiled":
                    warmup_count = (
                        args.compile_warmups
                    )
                else:
                    warmup_count = (
                        args.eager_warmups
                    )

                # ----------------------------------------------
                # Warmup
                # ----------------------------------------------

                for warmup_idx in range(
                    warmup_count
                ):
                    print(
                        f"Warmup "
                        f"{warmup_idx + 1}/"
                        f"{warmup_count}..."
                    )

                    try:
                        result = run_once(
                            model=model,
                            inputs=inputs,
                            mode=mode,
                            new_tokens=args.new_tokens,
                            compile_config=compile_config,
                        )

                        same = ""

                        if dynamic_reference is not None:
                            same = torch.equal(
                                result["tokens"],
                                dynamic_reference,
                            )

                        row = {
                            "model":
                                args.model,
                            "gpu":
                                torch.cuda.get_device_name(0),
                            "gpu_capability":
                                str(
                                    torch.cuda
                                    .get_device_capability(0)
                                ),
                            "torch_version":
                                torch.__version__,
                            "torch_cuda":
                                torch.version.cuda,
                            "transformers_version":
                                transformers.__version__,
                            "python_version":
                                platform.python_version(),
                            "hostname":
                                hostname,
                            "slurm_job_id":
                                slurm_job_id,
                            "git_commit":
                                git_commit,

                            "context_len":
                                context_len,
                            "mode":
                                mode,
                            "phase":
                                "warmup",
                            "repeat":
                                warmup_idx,

                            "generated_tokens":
                                result[
                                    "generated_tokens"
                                ],
                            "elapsed_sec":
                                result["elapsed_sec"],
                            "e2e_tok_s":
                                result["e2e_tok_s"],

                            "peak_allocated_gib":
                                result[
                                    "peak_allocated_gib"
                                ],
                            "peak_reserved_gib":
                                result[
                                    "peak_reserved_gib"
                                ],

                            "output_hash":
                                result["output_hash"],
                            "same_as_dynamic":
                                same,

                            "dynamo_unique_graphs":
                                result[
                                    "dynamo_unique_graphs"
                                ],
                            "dynamo_calls_captured":
                                result[
                                    "dynamo_calls_captured"
                                ],

                            "error":
                                "",
                        }

                        writer.writerow(row)
                        f.flush()

                        print(
                            f"time="
                            f"{result['elapsed_sec']:.3f}s "
                            f"tok/s="
                            f"{result['e2e_tok_s']:.2f} "
                            f"peak="
                            f"{result['peak_allocated_gib']:.3f}GiB "
                            f"hash="
                            f"{result['output_hash']} "
                            f"same="
                            f"{same} "
                            f"graphs="
                            f"{result['dynamo_unique_graphs']}"
                        )

                    except Exception as exc:
                        traceback.print_exc()

                        writer.writerow({
                            "model":
                                args.model,
                            "gpu":
                                torch.cuda.get_device_name(0),
                            "gpu_capability":
                                str(
                                    torch.cuda
                                    .get_device_capability(0)
                                ),
                            "torch_version":
                                torch.__version__,
                            "torch_cuda":
                                torch.version.cuda,
                            "transformers_version":
                                transformers.__version__,
                            "python_version":
                                platform.python_version(),
                            "hostname":
                                hostname,
                            "slurm_job_id":
                                slurm_job_id,
                            "git_commit":
                                git_commit,
                            "context_len":
                                context_len,
                            "mode":
                                mode,
                            "phase":
                                "warmup",
                            "repeat":
                                warmup_idx,
                            "error":
                                repr(exc),
                        })

                        f.flush()
                        raise

                # ----------------------------------------------
                # Measured runs
                # ----------------------------------------------

                for repeat in range(
                    args.repeats
                ):
                    print(
                        f"Run "
                        f"{repeat + 1}/"
                        f"{args.repeats}"
                    )

                    try:
                        result = run_once(
                            model=model,
                            inputs=inputs,
                            mode=mode,
                            new_tokens=args.new_tokens,
                            compile_config=compile_config,
                        )

                        if (
                            mode == "dynamic_eager"
                            and dynamic_reference is None
                        ):
                            dynamic_reference = (
                                result["tokens"].clone()
                            )

                        same = torch.equal(
                            result["tokens"],
                            dynamic_reference,
                        )

                        row = {
                            "model":
                                args.model,
                            "gpu":
                                torch.cuda.get_device_name(0),
                            "gpu_capability":
                                str(
                                    torch.cuda
                                    .get_device_capability(0)
                                ),
                            "torch_version":
                                torch.__version__,
                            "torch_cuda":
                                torch.version.cuda,
                            "transformers_version":
                                transformers.__version__,
                            "python_version":
                                platform.python_version(),
                            "hostname":
                                hostname,
                            "slurm_job_id":
                                slurm_job_id,
                            "git_commit":
                                git_commit,

                            "context_len":
                                context_len,
                            "mode":
                                mode,
                            "phase":
                                "measure",
                            "repeat":
                                repeat,

                            "generated_tokens":
                                result[
                                    "generated_tokens"
                                ],
                            "elapsed_sec":
                                result["elapsed_sec"],
                            "e2e_tok_s":
                                result["e2e_tok_s"],

                            "peak_allocated_gib":
                                result[
                                    "peak_allocated_gib"
                                ],
                            "peak_reserved_gib":
                                result[
                                    "peak_reserved_gib"
                                ],

                            "output_hash":
                                result["output_hash"],
                            "same_as_dynamic":
                                same,

                            "dynamo_unique_graphs":
                                result[
                                    "dynamo_unique_graphs"
                                ],
                            "dynamo_calls_captured":
                                result[
                                    "dynamo_calls_captured"
                                ],

                            "error":
                                "",
                        }

                        writer.writerow(row)
                        f.flush()

                        print(
                            f"time="
                            f"{result['elapsed_sec']:.3f}s "
                            f"tok/s="
                            f"{result['e2e_tok_s']:.2f} "
                            f"peak="
                            f"{result['peak_allocated_gib']:.3f}GiB "
                            f"hash="
                            f"{result['output_hash']} "
                            f"same="
                            f"{same} "
                            f"graphs="
                            f"{result['dynamo_unique_graphs']}"
                        )

                    except Exception as exc:
                        traceback.print_exc()

                        writer.writerow({
                            "model":
                                args.model,
                            "gpu":
                                torch.cuda.get_device_name(0),
                            "gpu_capability":
                                str(
                                    torch.cuda
                                    .get_device_capability(0)
                                ),
                            "torch_version":
                                torch.__version__,
                            "torch_cuda":
                                torch.version.cuda,
                            "transformers_version":
                                transformers.__version__,
                            "python_version":
                                platform.python_version(),
                            "hostname":
                                hostname,
                            "slurm_job_id":
                                slurm_job_id,
                            "git_commit":
                                git_commit,
                            "context_len":
                                context_len,
                            "mode":
                                mode,
                            "phase":
                                "measure",
                            "repeat":
                                repeat,
                            "error":
                                repr(exc),
                        })

                        f.flush()
                        raise

            del inputs
            gc.collect()
            torch.cuda.empty_cache()

    print()
    print("=" * 76)
    print("FINISHED")
    print("Results:", args.out)

    print(
        "Dynamo unique graphs:",
        dynamo_stat(
            "stats",
            "unique_graphs",
        ),
    )

    print(
        "Dynamo calls captured:",
        dynamo_stat(
            "stats",
            "calls_captured",
        ),
    )

    try:
        print(
            "Graph breaks:",
            dict(
                torch._dynamo
                .utils
                .counters["graph_break"]
            ),
        )
    except Exception:
        pass

    print("=" * 76)


if __name__ == "__main__":
    main()
