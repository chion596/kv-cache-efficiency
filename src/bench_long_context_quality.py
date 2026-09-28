import argparse
import csv
import gc
import hashlib
import math
import os
import random
import re
import time

import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    QuantizedCache,
)


def make_cache(
    model,
    cache_impl,
    q_group_size,
    residual_length,
):
    if cache_impl == "dynamic":
        return DynamicCache(
            config=model.config,
        )

    if cache_impl.startswith("int"):
        nbits = int(
            cache_impl[3:]
        )

        if nbits not in {
            1,
            2,
            3,
            4,
            8,
        }:
            raise ValueError(
                f"Unsupported HQQ nbits: {nbits}"
            )

        return QuantizedCache(
            backend="hqq",
            config=model.config,
            nbits=nbits,
            axis_key=1,
            axis_value=1,
            q_group_size=q_group_size,
            residual_length=residual_length,
        )

    raise ValueError(
        f"Unknown cache: {cache_impl}"
    )


ADJECTIVES = [
    "blue",
    "quiet",
    "northern",
    "silver",
    "small",
    "ancient",
    "green",
    "formal",
    "central",
    "wooden",
]

NOUNS = [
    "archive",
    "station",
    "document",
    "library",
    "garden",
    "office",
    "museum",
    "workshop",
    "laboratory",
    "warehouse",
]

PLACES = [
    "river",
    "courtyard",
    "hill",
    "bridge",
    "harbor",
    "square",
    "forest",
    "campus",
    "valley",
    "market",
]

TOPICS = [
    "weather",
    "transport",
    "inventory",
    "history",
    "maintenance",
    "education",
    "planning",
    "research",
    "shipping",
    "architecture",
]


def make_filler_ids(
    tokenizer,
    seed,
):
    """
    Детерминированный natural-ish haystack.

    В нём нет 8-значных кодов, чтобы secret code
    оставался уникальным объектом retrieval.
    """
    rng = random.Random(
        100_000 + seed
    )

    sentences = []

    for i in range(3000):
        adjective = rng.choice(
            ADJECTIVES
        )
        noun = rng.choice(
            NOUNS
        )
        place = rng.choice(
            PLACES
        )
        topic = rng.choice(
            TOPICS
        )

        section = rng.randint(
            1,
            999,
        )

        sentences.append(
            f"Archive entry {i % 3000}: "
            f"The {adjective} {noun} "
            f"near the {place} was reviewed "
            f"under section {section}. "
            f"The routine note concerned "
            f"{topic} and contained no "
            f"special instructions.\n"
        )

    text = "".join(sentences)

    ids = tokenizer(
        text,
        add_special_tokens=False,
    ).input_ids

    if len(ids) < 33000:
        raise RuntimeError(
            "Generated filler is too short: "
            f"{len(ids)} tokens"
        )

    return ids


def make_passkey(
    seed,
    context_len,
    position_index,
):
    derived_seed = (
        seed * 1_000_003
        + context_len * 97
        + position_index * 7919
    )

    rng = random.Random(
        derived_seed
    )

    return str(
        rng.randint(
            10_000_000,
            99_999_999,
        )
    )


def build_task(
    tokenizer,
    filler_ids,
    context_len,
    position,
    seed,
    position_index,
):
    passkey = make_passkey(
        seed,
        context_len,
        position_index,
    )

    needle_text = (
        "\nIMPORTANT MEMORY RECORD\n"
        f"The secret access code is {passkey}.\n"
        "Remember this exact eight-digit code.\n"
        "END IMPORTANT MEMORY RECORD\n"
    )

    query_text = (
        "\nQUESTION\n"
        "What is the secret access code from "
        "the IMPORTANT MEMORY RECORD? "
        "Reply with only the eight digits of "
        "the code, with no spaces or punctuation.\n"
        "ANSWER:"
    )

    needle_ids = tokenizer(
        needle_text,
        add_special_tokens=False,
    ).input_ids

    query_ids = tokenizer(
        query_text,
        add_special_tokens=False,
    ).input_ids

    body_budget = (
        context_len
        - len(needle_ids)
        - len(query_ids)
    )

    if body_budget <= 0:
        raise RuntimeError(
            "Context is too small for task"
        )

    if len(filler_ids) < body_budget:
        raise RuntimeError(
            "Filler is shorter than body budget"
        )

    body = filler_ids[:body_budget]

    insert_at = int(
        round(
            position * body_budget
        )
    )

    insert_at = max(
        0,
        min(
            insert_at,
            body_budget,
        ),
    )

    prompt_ids = (
        body[:insert_at]
        + needle_ids
        + body[insert_at:]
        + query_ids
    )

    assert (
        len(prompt_ids)
        == context_len
    ), (
        len(prompt_ids),
        context_len,
    )

    # Query заканчивается на "ANSWER:".
    # Пробел является частью правильного continuation.
    target_text = " " + passkey

    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
    ).input_ids

    return {
        "prompt_ids": prompt_ids,
        "target_ids": target_ids,
        "passkey": passkey,
        "needle_start_token": insert_at,
        "needle_position": position,
    }


def token_hash(ids):
    payload = ",".join(
        str(x) for x in ids
    ).encode()

    return hashlib.sha256(
        payload
    ).hexdigest()[:16]


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    prompt_ids,
    cache_impl,
    q_group_size,
    residual_length,
    max_new_tokens,
):
    device = (
        model.get_input_embeddings()
        .weight.device
    )

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    context_len = input_ids.shape[-1]

    cache = make_cache(
        model,
        cache_impl,
        q_group_size=q_group_size,
        residual_length=residual_length,
    )

    cache_position = torch.arange(
        context_len,
        dtype=torch.int64,
        device=device,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )

    cache = outputs.past_key_values

    next_token = (
        outputs.logits[:, -1, :]
        .argmax(dim=-1)
        .unsqueeze(-1)
    )

    generated = [
        int(next_token.item())
    ]

    del outputs

    if (
        tokenizer.eos_token_id
        is not None
        and generated[-1]
        == tokenizer.eos_token_id
    ):
        text = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        return generated, text

    current_position = torch.tensor(
        [context_len],
        dtype=torch.int64,
        device=device,
    )

    for _ in range(
        max_new_tokens - 1
    ):
        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_ones(
                    (1, 1)
                ),
            ],
            dim=-1,
        )

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            cache_position=current_position,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

        cache = outputs.past_key_values

        next_token = (
            outputs.logits[:, -1, :]
            .argmax(dim=-1)
            .unsqueeze(-1)
        )

        token_id = int(
            next_token.item()
        )

        generated.append(
            token_id
        )

        current_position = (
            current_position + 1
        )

        del outputs

        if (
            tokenizer.eos_token_id
            is not None
            and token_id
            == tokenizer.eos_token_id
        ):
            break

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    return generated, text


@torch.inference_mode()
def teacher_forced_score(
    model,
    prompt_ids,
    target_ids,
    cache_impl,
    q_group_size,
    residual_length,
):
    """
    Считает log-probability правильного ответа.

    Первый target token предсказывается сразу после prefill.
    Tokens 2..N уже требуют использования ранее сохранённого
    KV cache при decode, поэтому отдельно считаем decode-only
    метрики.
    """
    device = (
        model.get_input_embeddings()
        .weight.device
    )

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    context_len = input_ids.shape[-1]

    cache = make_cache(
        model,
        cache_impl,
        q_group_size=q_group_size,
        residual_length=residual_length,
    )

    cache_position = torch.arange(
        context_len,
        dtype=torch.int64,
        device=device,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )

    cache = outputs.past_key_values

    logits = outputs.logits[
        :, -1, :
    ]

    del outputs

    logprobs = []
    top1_correct = []

    for i, target_id in enumerate(
        target_ids
    ):
        log_probs = F.log_softmax(
            logits.float(),
            dim=-1,
        )

        lp = float(
            log_probs[
                0,
                target_id,
            ].item()
        )

        pred = int(
            logits.argmax(
                dim=-1
            ).item()
        )

        logprobs.append(
            lp
        )

        top1_correct.append(
            int(pred == target_id)
        )

        if i == len(target_ids) - 1:
            break

        current_token = torch.tensor(
            [[target_id]],
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_ones(
                    (1, 1)
                ),
            ],
            dim=-1,
        )

        position = torch.tensor(
            [context_len + i],
            dtype=torch.int64,
            device=device,
        )

        outputs = model(
            input_ids=current_token,
            attention_mask=attention_mask,
            cache_position=position,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

        cache = outputs.past_key_values

        logits = outputs.logits[
            :, -1, :
        ]

        del outputs

    all_nll = -sum(
        logprobs
    ) / len(logprobs)

    all_top1 = (
        100.0
        * sum(top1_correct)
        / len(top1_correct)
    )

    if len(logprobs) > 1:
        decode_lps = logprobs[1:]
        decode_top1 = top1_correct[1:]

        decode_nll = (
            -sum(decode_lps)
            / len(decode_lps)
        )

        decode_top1_pct = (
            100.0
            * sum(decode_top1)
            / len(decode_top1)
        )

        decode_mean_logprob = (
            sum(decode_lps)
            / len(decode_lps)
        )

    else:
        decode_nll = float("nan")
        decode_top1_pct = float("nan")
        decode_mean_logprob = float("nan")

    return {
        "target_token_count":
            len(target_ids),

        "tf_nll_all":
            all_nll,

        "tf_top1_all_pct":
            all_top1,

        "tf_nll_decode":
            decode_nll,

        "tf_mean_logprob_decode":
            decode_mean_logprob,

        "tf_top1_decode_pct":
            decode_top1_pct,
    }


def extract_digits(text):
    digits = "".join(
        re.findall(
            r"\d",
            text,
        )
    )

    return digits[:8]


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
            8192,
            16384,
            32768,
        ],
    )

    parser.add_argument(
        "--positions",
        nargs="+",
        type=float,
        default=[
            0.1,
            0.5,
            0.9,
        ],
    )

    parser.add_argument(
        "--caches",
        nargs="+",
        default=[
            "dynamic",
            "int4",
            "int2",
        ],
    )

    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--residual-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--device-map",
        choices=[
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ],
        default=None,
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    torch.manual_seed(42)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model,
            local_files_only=True,
        )
    )

    load_kwargs = {
        "dtype": torch.float16,
        "local_files_only": True,
    }

    if args.device_map is not None:
        load_kwargs[
            "device_map"
        ] = args.device_map

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                args.model,
                **load_kwargs,
            )
        )
    else:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                args.model,
                **load_kwargs,
            )
            .to("cuda")
        )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    print("=" * 72)
    print("LONG-CONTEXT KV CACHE QUALITY BENCHMARK")
    print("=" * 72)
    print("Model:", args.model)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("Contexts:", args.contexts)
    print("Positions:", args.positions)
    print("Caches:", args.caches)
    print("Q group size:", args.q_group_size)
    print("Residual length:", args.residual_length)
    print("Device map:", args.device_map)

    if hasattr(
        model,
        "hf_device_map",
    ):
        print(
            "HF device map:",
            model.hf_device_map,
        )

    print("Seeds:", args.num_seeds)
    print("=" * 72)

    filler_cache = {}

    for seed in range(
        args.num_seeds
    ):
        print(
            f"Building filler for seed {seed}..."
        )

        filler_cache[seed] = (
            make_filler_ids(
                tokenizer,
                seed,
            )
        )

        print(
            "  tokens:",
            len(filler_cache[seed]),
        )

    out_dir = os.path.dirname(
        args.out
    )

    if out_dir:
        os.makedirs(
            out_dir,
            exist_ok=True,
        )

    fields = [
        "model",
        "cache",
        "q_group_size",
        "residual_length",
        "context_len",
        "position",
        "seed",
        "passkey",
        "needle_start_token",

        "target_token_count",

        "generated_text",
        "generated_hash",
        "normalized_answer",

        "strict_exact_match",
        "normalized_exact_match",

        "tf_nll_all",
        "tf_top1_all_pct",

        "tf_nll_decode",
        "tf_mean_logprob_decode",
        "tf_top1_decode_pct",

        "elapsed_sec",
        "error",
    ]

    all_rows = []

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
            print()
            print("=" * 72)
            print(
                "CACHE:",
                cache_impl,
            )
            print("=" * 72)

            for context_len in args.contexts:
                for position_index, position in enumerate(
                    args.positions
                ):
                    for seed in range(
                        args.num_seeds
                    ):
                        task = build_task(
                            tokenizer=tokenizer,
                            filler_ids=filler_cache[
                                seed
                            ],
                            context_len=context_len,
                            position=position,
                            seed=seed,
                            position_index=(
                                position_index
                            ),
                        )

                        print(
                            f"cache={cache_impl} "
                            f"context={context_len} "
                            f"position={position:.1f} "
                            f"seed={seed} "
                            f"code={task['passkey']}"
                        )

                        start = time.perf_counter()

                        try:
                            gc.collect()
                            torch.cuda.empty_cache()

                            generated_ids, generated_text = (
                                greedy_generate(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt_ids=task[
                                        "prompt_ids"
                                    ],
                                    cache_impl=cache_impl,
                                    q_group_size=(
                                        args.q_group_size
                                    ),
                                    residual_length=(
                                        args.residual_length
                                    ),
                                    max_new_tokens=(
                                        args.max_new_tokens
                                    ),
                                )
                            )

                            normalized = (
                                extract_digits(
                                    generated_text
                                )
                            )

                            strict_em = int(
                                generated_text.strip()
                                == task["passkey"]
                            )

                            normalized_em = int(
                                normalized
                                == task["passkey"]
                            )

                            gc.collect()
                            torch.cuda.empty_cache()

                            tf = (
                                teacher_forced_score(
                                    model=model,
                                    prompt_ids=task[
                                        "prompt_ids"
                                    ],
                                    target_ids=task[
                                        "target_ids"
                                    ],
                                    cache_impl=(
                                        cache_impl
                                    ),
                                    q_group_size=(
                                        args.q_group_size
                                    ),
                                    residual_length=(
                                        args.residual_length
                                    ),
                                )
                            )

                            elapsed = (
                                time.perf_counter()
                                - start
                            )

                            row = {
                                "model":
                                    args.model,

                                "cache":
                                    cache_impl,

                                "q_group_size":
                                    args.q_group_size,

                                "residual_length":
                                    args.residual_length,

                                "context_len":
                                    context_len,

                                "position":
                                    position,

                                "seed":
                                    seed,

                                "passkey":
                                    task[
                                        "passkey"
                                    ],

                                "needle_start_token":
                                    task[
                                        "needle_start_token"
                                    ],

                                **tf,

                                "generated_text":
                                    generated_text,

                                "generated_hash":
                                    token_hash(
                                        generated_ids
                                    ),

                                "normalized_answer":
                                    normalized,

                                "strict_exact_match":
                                    strict_em,

                                "normalized_exact_match":
                                    normalized_em,

                                "elapsed_sec":
                                    elapsed,

                                "error":
                                    "",
                            }

                            writer.writerow(
                                row
                            )

                            f.flush()

                            all_rows.append(
                                row
                            )

                            print(
                                f"  generated="
                                f"{generated_text!r} "
                                f"EM={normalized_em} "
                                f"TF-decode-NLL="
                                f"{tf['tf_nll_decode']:.4f} "
                                f"TF-decode-top1="
                                f"{tf['tf_top1_decode_pct']:.1f}%"
                            )

                        except Exception as exc:
                            import traceback
                            traceback.print_exc()

                            row = {
                                "model":
                                    args.model,

                                "cache":
                                    cache_impl,

                                "q_group_size":
                                    args.q_group_size,

                                "residual_length":
                                    args.residual_length,

                                "context_len":
                                    context_len,

                                "position":
                                    position,

                                "seed":
                                    seed,

                                "passkey":
                                    task[
                                        "passkey"
                                    ],

                                "needle_start_token":
                                    task[
                                        "needle_start_token"
                                    ],

                                "error":
                                    repr(exc),
                            }

                            writer.writerow(
                                row
                            )
                            f.flush()

                            all_rows.append(
                                row
                            )

    print()
    print("=" * 72)
    print("FINISHED")
    print("Results:", args.out)
    print("=" * 72)

    valid = [
        row
        for row in all_rows
        if not row.get(
            "error"
        )
    ]

    for cache_impl in args.caches:
        rows = [
            r
            for r in valid
            if r["cache"]
            == cache_impl
        ]

        if not rows:
            continue

        accuracy = (
            100.0
            * sum(
                r[
                    "normalized_exact_match"
                ]
                for r in rows
            )
            / len(rows)
        )

        tf_top1 = sum(
            r[
                "tf_top1_decode_pct"
            ]
            for r in rows
        ) / len(rows)

        tf_nll = sum(
            r[
                "tf_nll_decode"
            ]
            for r in rows
        ) / len(rows)

        print(
            f"{cache_impl}: "
            f"n={len(rows)} "
            f"retrieval_accuracy="
            f"{accuracy:.2f}% "
            f"tf_decode_top1="
            f"{tf_top1:.2f}% "
            f"tf_decode_nll="
            f"{tf_nll:.4f}"
        )


if __name__ == "__main__":
    main()
