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

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def token_hash(ids):
    payload = ",".join(
        str(x) for x in ids.tolist()
    ).encode()

    return hashlib.sha256(payload).hexdigest()[:16]


def first_difference(a, b):
    n = min(len(a), len(b))

    for i in range(n):
        if a[i].item() != b[i].item():
            return i

    if len(a) != len(b):
        return n

    return None


@torch.inference_mode()
def generate(
    model,
    inputs,
    cache_impl,
    new_tokens,
):
    output = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        cache_implementation=cache_impl,
        disable_compile=True,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        pad_token_id=model.generation_config.pad_token_id,
    )

    context_len = inputs["input_ids"].shape[-1]

    return output[0, context_len:].cpu()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)

    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[512, 2048, 8192],
    )

    parser.add_argument(
        "--new-tokens",
        type=int,
        default=32,
    )

    args = parser.parse_args()

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

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("GPU:", torch.cuda.get_device_name(0))
    print("Attention implementation:",
          getattr(model.config, "_attn_implementation", "unknown"))

    caches = [
        "dynamic",
        "static",
        "offloaded",
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

        outputs = {}

        for cache_impl in caches:
            # Запускаем дважды, чтобы одновременно проверить
            # воспроизводимость одной и той же стратегии.
            runs = []

            for repeat in range(2):
                torch.cuda.empty_cache()

                ids = generate(
                    model,
                    inputs,
                    cache_impl,
                    args.new_tokens,
                )

                runs.append(ids)

                print(
                    cache_impl,
                    "repeat=",
                    repeat,
                    "hash=",
                    token_hash(ids),
                    "ids=",
                    ids.tolist(),
                )

            print(
                cache_impl,
                "self-consistent:",
                torch.equal(
                    runs[0],
                    runs[1],
                ),
            )

            outputs[cache_impl] = runs[0]

        reference = outputs["dynamic"]

        for cache_impl in [
            "static",
            "offloaded",
        ]:
            candidate = outputs[cache_impl]

            same = torch.equal(
                reference,
                candidate,
            )

            print()
            print(
                f"dynamic == {cache_impl}:",
                same,
            )

            if not same:
                pos = first_difference(
                    reference,
                    candidate,
                )

                print(
                    "first differing generated token:",
                    pos,
                )

                if pos is not None:
                    print(
                        "dynamic token:",
                        reference[pos].item(),
                    )
                    print(
                        f"{cache_impl} token:",
                        candidate[pos].item(),
                    )

        print()
        print(
            "dynamic text:",
            tokenizer.decode(
                reference,
                skip_special_tokens=True,
            ),
        )

        print(
            "offloaded text:",
            tokenizer.decode(
                outputs["offloaded"],
                skip_special_tokens=True,
            ),
        )


if __name__ == "__main__":
    main()
