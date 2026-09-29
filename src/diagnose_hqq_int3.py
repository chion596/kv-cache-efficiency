import gc
import traceback

import torch

from hqq.core.quantize import Quantizer as HQQQuantizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    QuantizedCache,
)


def direct_hqq_test(
    group_size,
    seq_len=512,
):
    print()
    print("=" * 80)
    print(
        f"DIRECT HQQ INT3 "
        f"group={group_size} "
        f"seq={seq_len}"
    )
    print("=" * 80)

    # Форма одного KV tensor для Qwen:
    # [batch, kv_heads, seq, head_dim]
    x = torch.randn(
        1,
        8,
        seq_len,
        128,
        device="cuda",
        dtype=torch.float16,
    )

    try:
        q, meta = HQQQuantizer.quantize(
            x,
            axis=1,
            device="cuda",
            compute_dtype=torch.float16,
            nbits=3,
            group_size=group_size,
        )

        HQQQuantizer.cuda(
            q,
            meta=meta,
            device="cuda",
        )

        meta["scale"] = meta["scale"].to(
            q.device
        )
        meta["zero"] = meta["zero"].to(
            q.device
        )
        meta["compute_dtype"] = (
            torch.float16
        )

        print(
            "quantized shape:",
            q.shape,
        )
        print(
            "quantized dtype:",
            q.dtype,
        )
        print(
            "scale shape:",
            meta["scale"].shape,
        )
        print(
            "zero shape:",
            meta["zero"].shape,
        )

        y = HQQQuantizer.dequantize(
            q,
            meta,
        )

        print(
            "dequantized:",
            y.shape,
            y.dtype,
        )

        mae = (
            (x - y)
            .abs()
            .mean()
            .item()
        )

        print(
            f"mean abs error: {mae:.6f}"
        )
        print("RESULT: PASS")

    except Exception:
        print("RESULT: FAIL")
        traceback.print_exc()

    del x
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def cache_test(
    model,
    tokenizer,
    group_size,
    seq_len=512,
):
    print()
    print("=" * 80)
    print(
        f"TRANSFORMERS QUANTIZEDCACHE INT3 "
        f"group={group_size} "
        f"seq={seq_len}"
    )
    print("=" * 80)

    text = (
        "KV cache quantization diagnostic. "
    )

    ids = tokenizer(
        text,
        add_special_tokens=False,
    ).input_ids

    ids = (
        ids
        * (
            seq_len // len(ids)
            + 1
        )
    )[:seq_len]

    input_ids = torch.tensor(
        [ids],
        device="cuda",
        dtype=torch.long,
    )

    mask = torch.ones_like(
        input_ids
    )

    try:
        cache = QuantizedCache(
            backend="hqq",
            config=model.config,
            nbits=3,
            axis_key=1,
            axis_value=1,
            q_group_size=group_size,
            residual_length=128,
        )

        pos = torch.arange(
            seq_len,
            device="cuda",
        )

        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            cache_position=pos,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

        cache = out.past_key_values

        # Главное: заставляем cache пройти
        # dequantization на следующем decode step.
        token = (
            out.logits[:, -1:, :]
            .argmax(dim=-1)
        )

        mask = torch.cat(
            [
                mask,
                torch.ones(
                    (1, 1),
                    device="cuda",
                    dtype=mask.dtype,
                ),
            ],
            dim=-1,
        )

        out = model(
            input_ids=token,
            attention_mask=mask,
            cache_position=torch.tensor(
                [seq_len],
                device="cuda",
            ),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

        print(
            "cache seq len:",
            out.past_key_values
            .get_seq_length()
        )

        print("RESULT: PASS")

    except Exception:
        print("RESULT: FAIL")
        traceback.print_exc()

    gc.collect()
    torch.cuda.empty_cache()


def main():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    model_path = (
        "/scratch/ws/"
        "nashilovskiy-kv-cache/"
        "models/Qwen3-1.7B"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_path,
            dtype=torch.float16,
            local_files_only=True,
        )
        .to("cuda")
    )

    model.eval()

    for group in [
        16,
        32,
        64,
        128,
    ]:
        direct_hqq_test(
            group_size=group,
        )

    for group in [
        16,
        32,
        64,
        128,
    ]:
        cache_test(
            model=model,
            tokenizer=tokenizer,
            group_size=group,
        )


if __name__ == "__main__":
    main()
