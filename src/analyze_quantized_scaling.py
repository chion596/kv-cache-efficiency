from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FILES = {
    "0.6B": (
        "results/raw/"
        "full_quantized_hqq_qwen3_0.6b_4355154.csv"
    ),
    "1.7B": (
        "results/raw/"
        "full_quantized_hqq_qwen3_1.7b_4355206.csv"
    ),
    "4B": (
        "results/raw/"
        "full_quantized_hqq_qwen3_4b_4355281.csv"
    ),
}

OUT = Path("results/summary/quantized_scaling")
PLOTS = Path("plots/quantized_scaling")

OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


def load_model(label, path):
    df = pd.read_csv(path)

    errors = (
        df["error"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[errors == ""].copy()
    df["model_size"] = label

    return df


frames = [
    load_model(label, path)
    for label, path in FILES.items()
]

df = pd.concat(
    frames,
    ignore_index=True,
)

summary = (
    df.groupby(
        [
            "model_size",
            "context_len",
            "cache",
        ],
        as_index=False,
    )
    .agg(
        n=("repeat", "count"),

        model_memory_gib=(
            "model_memory_gib",
            "mean",
        ),

        prefill_sec=(
            "prefill_sec",
            "mean",
        ),

        decode_tok_s=(
            "decode_tok_s",
            "mean",
        ),

        tpot_ms=(
            "tpot_ms",
            "mean",
        ),

        e2e_sec=(
            "e2e_sec",
            "mean",
        ),

        peak_gpu_gib=(
            "total_peak_allocated_gib",
            "mean",
        ),

        kv_gpu_gib=(
            "cache_prompt_gpu_gib",
            "mean",
        ),

        fp16_kv_theory_gib=(
            "theoretical_kv_prompt_gib",
            "mean",
        ),

        hash_count=(
            "output_hash",
            "nunique",
        ),
    )
)

summary.to_csv(
    OUT / "cross_model_summary.csv",
    index=False,
)

rows = []

for model in summary.model_size.unique():
    m = summary[
        summary.model_size == model
    ]

    for context in sorted(
        m.context_len.unique()
    ):
        dyn_rows = m[
            (m.context_len == context)
            & (m.cache == "dynamic")
        ]

        if len(dyn_rows) == 0:
            continue

        dyn = dyn_rows.iloc[0]

        for cache in ["int4", "int2"]:
            q_rows = m[
                (m.context_len == context)
                & (m.cache == cache)
            ]

            if len(q_rows) == 0:
                continue

            q = q_rows.iloc[0]

            rows.append({
                "model_size": model,
                "context_len": context,
                "cache": cache,

                "model_memory_gib":
                    dyn.model_memory_gib,

                "dynamic_peak_gib":
                    dyn.peak_gpu_gib,

                "quant_peak_gib":
                    q.peak_gpu_gib,

                "absolute_peak_saving_gib":
                    dyn.peak_gpu_gib
                    - q.peak_gpu_gib,

                "total_peak_saving_pct":
                    100 * (
                        1
                        - q.peak_gpu_gib
                        / dyn.peak_gpu_gib
                    ),

                "dynamic_kv_gib":
                    dyn.kv_gpu_gib,

                "quant_kv_gib":
                    q.kv_gpu_gib,

                "kv_saving_pct":
                    100 * (
                        1
                        - q.kv_gpu_gib
                        / dyn.kv_gpu_gib
                    ),

                "kv_compression_x":
                    dyn.kv_gpu_gib
                    / q.kv_gpu_gib,

                "prefill_penalty_pct":
                    100 * (
                        q.prefill_sec
                        / dyn.prefill_sec
                        - 1
                    ),

                "decode_slowdown_pct":
                    100 * (
                        1
                        - q.decode_tok_s
                        / dyn.decode_tok_s
                    ),

                "e2e_penalty_pct":
                    100 * (
                        q.e2e_sec
                        / dyn.e2e_sec
                        - 1
                    ),
            })

comparison = pd.DataFrame(rows)

comparison.to_csv(
    OUT / "cross_model_vs_dynamic.csv",
    index=False,
)

print("=" * 120)
print("CROSS-MODEL: CONTEXT ~40K")
print("=" * 120)

at_40k = comparison[
    comparison.context_len == 40895
]

print(
    at_40k[
        [
            "model_size",
            "cache",
            "model_memory_gib",
            "dynamic_peak_gib",
            "quant_peak_gib",
            "absolute_peak_saving_gib",
            "total_peak_saving_pct",
            "kv_compression_x",
            "decode_slowdown_pct",
            "e2e_penalty_pct",
        ]
    ]
    .round(3)
    .to_string(index=False)
)

# ------------------------------------------------------------
# Plot 1: total GPU peak
# ------------------------------------------------------------

for model in [
    "0.6B",
    "1.7B",
    "4B",
]:
    sub = summary[
        summary.model_size == model
    ]

    for cache in [
        "dynamic",
        "int4",
        "int2",
    ]:
        s = sub[
            sub.cache == cache
        ].sort_values("context_len")

        plt.plot(
            s.context_len,
            s.peak_gpu_gib,
            marker="o",
            label=f"{model} {cache}",
        )

plt.xlabel("Context length")
plt.ylabel("Peak allocated GPU memory, GiB")
plt.title("GPU memory vs context")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS / "gpu_peak_vs_context.png",
    dpi=180,
)

plt.close()

# ------------------------------------------------------------
# Plot 2: total memory saving
# ------------------------------------------------------------

for model in [
    "0.6B",
    "1.7B",
    "4B",
]:
    for cache in [
        "int4",
        "int2",
    ]:
        s = comparison[
            (comparison.model_size == model)
            & (comparison.cache == cache)
        ].sort_values("context_len")

        plt.plot(
            s.context_len,
            s.total_peak_saving_pct,
            marker="o",
            label=f"{model} {cache}",
        )

plt.xlabel("Context length")
plt.ylabel("Total GPU peak saving, %")
plt.title("Memory benefit of KV quantization")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS / "memory_saving_vs_context.png",
    dpi=180,
)

plt.close()

# ------------------------------------------------------------
# Plot 3: absolute GiB saving
# ------------------------------------------------------------

for model in [
    "0.6B",
    "1.7B",
    "4B",
]:
    for cache in [
        "int4",
        "int2",
    ]:
        s = comparison[
            (comparison.model_size == model)
            & (comparison.cache == cache)
        ].sort_values("context_len")

        plt.plot(
            s.context_len,
            s.absolute_peak_saving_gib,
            marker="o",
            label=f"{model} {cache}",
        )

plt.xlabel("Context length")
plt.ylabel("Absolute GPU peak saving, GiB")
plt.title("Absolute GPU memory saved")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS / "absolute_memory_saving.png",
    dpi=180,
)

plt.close()

# ------------------------------------------------------------
# Plot 4: decode slowdown
# ------------------------------------------------------------

for model in [
    "0.6B",
    "1.7B",
    "4B",
]:
    for cache in [
        "int4",
        "int2",
    ]:
        s = comparison[
            (comparison.model_size == model)
            & (comparison.cache == cache)
        ].sort_values("context_len")

        plt.plot(
            s.context_len,
            s.decode_slowdown_pct,
            marker="o",
            label=f"{model} {cache}",
        )

plt.xlabel("Context length")
plt.ylabel("Decode throughput slowdown, %")
plt.title("Decode cost of KV quantization")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS / "decode_slowdown_vs_context.png",
    dpi=180,
)

plt.close()

print()
print("Saved:")
print(
    OUT / "cross_model_summary.csv"
)
print(
    OUT / "cross_model_vs_dynamic.csv"
)
print("Plots:", PLOTS)
