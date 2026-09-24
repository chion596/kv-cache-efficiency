import argparse
import hashlib
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    s = ",".join(map(str, ids.tolist())).encode()
    return hashlib.sha256(s).hexdigest()[:16]


@torch.inference_mode()
def run(model, inputs, cache_impl, new_tokens):
    out = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        cache_implementation=cache_impl,
        disable_compile=True,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        pad_token_id=model.generation_config.pad_token_id,
    )

    n = inputs["input_ids"].shape[-1]
    return out[0, n:].cpu()


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument("--attn", default="sdpa")
    p.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[512, 1024, 1536, 2048, 4096],
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--new-tokens", type=int, default=32)

    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation=args.attn,
        local_files_only=True,
    ).to("cuda")

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("GPU:", torch.cuda.get_device_name(0))
    print("Attention:", args.attn)

    caches = [
        "dynamic",
        "offloaded",
        "offloaded_static",
    ]

    for context_len in args.contexts:
        print()
        print("=" * 70)
        print("CONTEXT:", context_len)
        print("=" * 70)

        inputs = make_input(
            tokenizer,
            context_len,
            "cuda",
        )

        reference = run(
            model,
            inputs,
            "dynamic",
            args.new_tokens,
        )

        print(
            "dynamic reference:",
            token_hash(reference),
        )

        for cache_impl in caches:
            hashes = []

            for r in range(args.repeats):
                torch.cuda.empty_cache()

                ids = run(
                    model,
                    inputs,
                    cache_impl,
                    args.new_tokens,
                )

                h = token_hash(ids)
                hashes.append(h)

                print(
                    f"{cache_impl:16s} "
                    f"repeat={r} "
                    f"hash={h} "
                    f"same_as_dynamic={torch.equal(ids, reference)}"
                )

            print(
                f"{cache_impl:16s} unique_hashes="
                f"{len(set(hashes))}"
            )


if __name__ == "__main__":
    main()
