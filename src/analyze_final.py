from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

RAW = Path("results/raw")
OUT = Path("results/summary/final")
PLOTS = Path("plots/final")

PLOTS.mkdir(
    parents=True,
    exist_ok=True,
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPER
# ============================================================

def load_good(path):
    """
    Читаем CSV и оставляем только успешные строки.

    Некоторые benchmark-скрипты не падают полностью при ошибке,
    а записывают текст ошибки в колонку "error".
    """
    df = pd.read_csv(path)

    if "error" in df.columns:
        errors = (
            df["error"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df = df[
            errors == ""
        ].copy()

    return df


# ============================================================
# PART A1: QUALITY SCALING
# 0.6B -> 1.7B -> 4B -> 8B -> 14B
# ============================================================

QUALITY_FILES = {
    "0.6B": RAW / "quality_hqq_qwen3_0.6b_4355340.csv",
    "1.7B": RAW / "quality_hqq_qwen3_1.7b_4358632.csv",
    "4B": RAW / "quality_hqq_qwen3_4b_4358633.csv",
    "8B": RAW / "quality_hqq_qwen3_8b_4357836.csv",
    "14B": RAW / "quality_hqq_qwen3_14b_4359064.csv",
}


quality_rows = []

for model_size, path in QUALITY_FILES.items():

    df = load_good(path)

    for cache, group in df.groupby("cache"):

        quality_rows.append({
            "model_size": model_size,
            "cache": cache,
            "n": len(group),

            "retrieval_accuracy_pct":
                100
                * group[
                    "normalized_exact_match"
                ].mean(),

            "tf_decode_top1_pct":
                group[
                    "tf_top1_decode_pct"
                ].mean(),

            "tf_decode_nll":
                group[
                    "tf_nll_decode"
                ].mean(),
        })


quality = pd.DataFrame(
    quality_rows
)


MODEL_ORDER = {
    "0.6B": 0,
    "1.7B": 1,
    "4B": 2,
    "8B": 3,
    "14B": 4,
}


quality["_model_order"] = (
    quality["model_size"]
    .map(MODEL_ORDER)
)

quality = (
    quality
    .sort_values(
        [
            "_model_order",
            "cache",
        ]
    )
    .drop(
        columns="_model_order"
    )
)


quality.to_csv(
    OUT / "model_scaling_quality.csv",
    index=False,
)


# ============================================================
# PART A2: PERFORMANCE SCALING
# 0.6B -> 1.7B -> 4B -> 8B
# ============================================================

PERFORMANCE_FILES = {
    "0.6B": RAW / "full_quantized_hqq_qwen3_0.6b_4355154.csv",
    "1.7B": RAW / "full_quantized_hqq_qwen3_1.7b_4355206.csv",
    "4B": RAW / "full_quantized_hqq_qwen3_4b_4355281.csv",
    "8B": RAW / "full_quantized_hqq_qwen3_8b_4357755.csv",
}


performance_frames = []

for model_size, path in PERFORMANCE_FILES.items():

    df = load_good(path)

    df["model_size"] = model_size

    performance_frames.append(
        df
    )


performance_raw = pd.concat(
    performance_frames,
    ignore_index=True,
)


performance = (
    performance_raw
    .groupby(
        [
            "model_size",
            "context_len",
            "cache",
        ],
        as_index=False,
    )
    .agg(
        n=(
            "repeat",
            "count",
        ),

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
    )
)


performance["_model_order"] = (
    performance["model_size"]
    .map(MODEL_ORDER)
)

performance = (
    performance
    .sort_values(
        [
            "_model_order",
            "context_len",
            "cache",
        ]
    )
    .drop(
        columns="_model_order"
    )
)


performance.to_csv(
    OUT / "model_scaling_performance.csv",
    index=False,
)

# ============================================================
# PART B1: GROUP-SIZE QUALITY
# QWEN3-1.7B
# ============================================================

GROUP_QUALITY_FILES = [
    ("int8", 64, RAW / "quality_ablation_qwen3_1.7b_int8_g64_4359062.csv"),

    ("int4", 16, RAW / "quality_ablation_qwen3_1.7b_int4_g16_4359062.csv"),
    ("int4", 32, RAW / "final_quality_ablation_qwen3_1.7b_int4_g32_4363381.csv"),
    ("int4", 64, RAW / "quality_hqq_qwen3_1.7b_4358632.csv"),
    ("int4", 128, RAW / "quality_ablation_qwen3_1.7b_int4_g128_4359062.csv"),

    ("int2", 16, RAW / "final_quality_ablation_qwen3_1.7b_int2_g16_4363381.csv"),
    ("int2", 32, RAW / "final_quality_ablation_qwen3_1.7b_int2_g32_4363381.csv"),
    ("int2", 64, RAW / "quality_hqq_qwen3_1.7b_4358632.csv"),
    ("int2", 128, RAW / "final_quality_ablation_qwen3_1.7b_int2_g128_4363381.csv"),
]


group_quality_rows = []


for cache, group_size, path in GROUP_QUALITY_FILES:

    df = load_good(path)

    # Основной quality-файл содержит сразу
    # dynamic / int4 / int2.
    # Поэтому для него выбираем нужный cache.
    if "cache" in df.columns:
        df = df[
            df["cache"] == cache
        ].copy()

    group_quality_rows.append({
        "cache": cache,
        "group_size": group_size,
        "n": len(df),

        "retrieval_accuracy_pct":
            100
            * df[
                "normalized_exact_match"
            ].mean(),

        "tf_decode_top1_pct":
            df[
                "tf_top1_decode_pct"
            ].mean(),

        "tf_decode_nll":
            df[
                "tf_nll_decode"
            ].mean(),
    })


group_quality = pd.DataFrame(
    group_quality_rows
).sort_values(
    [
        "cache",
        "group_size",
    ]
)


group_quality.to_csv(
    OUT / "group_size_quality.csv",
    index=False,
)


# ============================================================
# PART B2: GROUP-SIZE PERFORMANCE
# QWEN3-1.7B
# ============================================================

GROUP_PERFORMANCE_FILES = [
    ("dynamic", 0, RAW / "final_perf_ablation_qwen3_1.7b_dynamic_4363381.csv"),

    ("int8", 64, RAW / "final_perf_ablation_qwen3_1.7b_int8_g64_4363381.csv"),

    ("int4", 16, RAW / "final_perf_ablation_qwen3_1.7b_int4_g16_4363381.csv"),
    ("int4", 32, RAW / "final_perf_ablation_qwen3_1.7b_int4_g32_4363381.csv"),
    ("int4", 64, RAW / "final_perf_ablation_qwen3_1.7b_int4_g64_4363381.csv"),
    ("int4", 128, RAW / "final_perf_ablation_qwen3_1.7b_int4_g128_4363381.csv"),

    ("int2", 16, RAW / "final_perf_ablation_qwen3_1.7b_int2_g16_4363381.csv"),
    ("int2", 32, RAW / "final_perf_ablation_qwen3_1.7b_int2_g32_4363381.csv"),
    ("int2", 64, RAW / "final_perf_ablation_qwen3_1.7b_int2_g64_4363381.csv"),
    ("int2", 128, RAW / "final_perf_ablation_qwen3_1.7b_int2_g128_4363381.csv"),
]


group_perf_rows = []


for cache, group_size, path in GROUP_PERFORMANCE_FILES:

    df = load_good(path)

    # Сейчас нас интересует самая длинная
    # общая точка ~41K.
    df = df[
        df["context_len"] == 40895
    ].copy()

    group_perf_rows.append({
        "cache": cache,
        "group_size": group_size,
        "n": len(df),

        "peak_gpu_gib":
            df[
                "total_peak_allocated_gib"
            ].mean(),

        "kv_gpu_gib":
            df[
                "cache_prompt_gpu_gib"
            ].mean(),

        "prefill_sec":
            df[
                "prefill_sec"
            ].mean(),

        "decode_tok_s":
            df[
                "decode_tok_s"
            ].mean(),

        "e2e_sec":
            df[
                "e2e_sec"
            ].mean(),
    })


group_performance = pd.DataFrame(
    group_perf_rows
).sort_values(
    [
        "cache",
        "group_size",
    ]
)


group_performance.to_csv(
    OUT / "group_size_performance.csv",
    index=False,
)

# ============================================================
# PART C: SHORT-CONTEXT CONTROL
# QWEN3-0.6B
# ============================================================

short_quality = load_good(
    RAW / "final_quality_short_qwen3_0.6b_4363382.csv"
)


short_context = (
    short_quality
    .groupby(
        [
            "cache",
            "context_len",
        ],
        as_index=False,
    )
    .agg(
        n=(
            "normalized_exact_match",
            "count",
        ),

        retrieval_accuracy_pct=(
            "normalized_exact_match",
            lambda x: 100 * x.mean(),
        ),

        tf_decode_top1_pct=(
            "tf_top1_decode_pct",
            "mean",
        ),

        tf_decode_nll=(
            "tf_nll_decode",
            "mean",
        ),
    )
    .sort_values(
        [
            "context_len",
            "cache",
        ]
    )
)


short_context.to_csv(
    OUT / "short_context_control.csv",
    index=False,
)

# ============================================================
# PART D: QUALITY / MEMORY / LATENCY TRADE-OFF
# QWEN3-1.7B AT ~41K
# ============================================================

dynamic_perf = (
    group_performance[
        group_performance["cache"] == "dynamic"
    ]
    .iloc[0]
)

dynamic_quality = (
    quality[
        (quality["model_size"] == "1.7B")
        & (quality["cache"] == "dynamic")
    ]
    .iloc[0]
)


tradeoff_rows = [
    {
        "cache": "dynamic",
        "group_size": 0,

        "retrieval_accuracy_pct":
            dynamic_quality[
                "retrieval_accuracy_pct"
            ],

        "kv_gpu_gib":
            dynamic_perf[
                "kv_gpu_gib"
            ],

        "kv_compression_x":
            1.0,

        "peak_gpu_gib":
            dynamic_perf[
                "peak_gpu_gib"
            ],

        "total_peak_saving_pct":
            0.0,

        "decode_tok_s":
            dynamic_perf[
                "decode_tok_s"
            ],

        "decode_slowdown_pct":
            0.0,
    }
]


for _, qrow in group_quality.iterrows():

    cache = qrow["cache"]
    group_size = qrow["group_size"]

    prow = group_performance[
        (group_performance["cache"] == cache)
        & (
            group_performance["group_size"]
            == group_size
        )
    ]

    if len(prow) == 0:
        continue

    prow = prow.iloc[0]

    tradeoff_rows.append({
        "cache": cache,
        "group_size": group_size,

        "retrieval_accuracy_pct":
            qrow[
                "retrieval_accuracy_pct"
            ],

        "kv_gpu_gib":
            prow[
                "kv_gpu_gib"
            ],

        "kv_compression_x":
            dynamic_perf[
                "kv_gpu_gib"
            ]
            / prow[
                "kv_gpu_gib"
            ],

        "peak_gpu_gib":
            prow[
                "peak_gpu_gib"
            ],

        "total_peak_saving_pct":
            100 * (
                1
                - prow[
                    "peak_gpu_gib"
                ]
                / dynamic_perf[
                    "peak_gpu_gib"
                ]
            ),

        "decode_tok_s":
            prow[
                "decode_tok_s"
            ],

        "decode_slowdown_pct":
            100 * (
                1
                - prow[
                    "decode_tok_s"
                ]
                / dynamic_perf[
                    "decode_tok_s"
                ]
            ),
    })


tradeoff = pd.DataFrame(
    tradeoff_rows
)


tradeoff.to_csv(
    OUT / "memory_latency_tradeoff.csv",
    index=False,
)

# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 90)
print("MODEL SCALING: QUALITY")
print("=" * 90)

print(
    quality
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 90)
print("MODEL SCALING: PERFORMANCE AT LONGEST CONTEXT")
print("=" * 90)


longest_rows = []

for model_size in [
    "0.6B",
    "1.7B",
    "4B",
    "8B",
]:

    sub = performance[
        performance["model_size"]
        == model_size
    ]

    max_context = (
        sub["context_len"]
        .max()
    )

    longest_rows.append(
        sub[
            sub["context_len"]
            == max_context
        ]
    )


longest = pd.concat(
    longest_rows,
    ignore_index=True,
)


print(
    longest[
        [
            "model_size",
            "context_len",
            "cache",
            "n",
            "peak_gpu_gib",
            "kv_gpu_gib",
            "prefill_sec",
            "decode_tok_s",
            "e2e_sec",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 90)
print("QWEN3-1.7B: GROUP-SIZE QUALITY")
print("=" * 90)

print(
    group_quality
    .round(4)
    .to_string(
        index=False
    )
)

print()
print("=" * 90)
print("QWEN3-1.7B: GROUP-SIZE PERFORMANCE AT 40895")
print("=" * 90)

print(
    group_performance
    .round(4)
    .to_string(
        index=False
    )
)

print()
print("=" * 90)
print("QWEN3-0.6B: SHORT-CONTEXT CONTROL")
print("=" * 90)

print(
    short_context
    .round(4)
    .to_string(
        index=False
    )
)

print()
print("=" * 90)
print("QWEN3-1.7B: QUALITY / MEMORY / LATENCY TRADE-OFF")
print("=" * 90)

print(
    tradeoff[
        [
            "cache",
            "group_size",
            "retrieval_accuracy_pct",
            "kv_compression_x",
            "peak_gpu_gib",
            "total_peak_saving_pct",
            "decode_tok_s",
            "decode_slowdown_pct",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)

print()
print("Saved final summaries:")
print(OUT / "model_scaling_quality.csv")
print(OUT / "model_scaling_performance.csv")
print(OUT / "group_size_quality.csv")
print(OUT / "group_size_performance.csv")
print(OUT / "short_context_control.csv")
print(OUT / "memory_latency_tradeoff.csv")

# ============================================================
# FIGURE 1: QUALITY VS MODEL SIZE
# ============================================================

model_labels = [
    "0.6B",
    "1.7B",
    "4B",
    "8B",
    "14B",
]

x = list(
    range(
        len(model_labels)
    )
)

for cache in [
    "dynamic",
    "int4",
    "int2",
]:
    sub = (
        quality[
            quality["cache"] == cache
        ]
        .set_index("model_size")
        .loc[model_labels]
        .reset_index()
    )

    plt.plot(
        x,
        sub["retrieval_accuracy_pct"],
        marker="o",
        label=cache,
    )

plt.xticks(
    x,
    model_labels,
)

plt.ylim(
    -5,
    105,
)

plt.xlabel(
    "Qwen3 model size"
)

plt.ylabel(
    "Retrieval exact-match, %"
)

plt.title(
    "Retrieval quality vs model size (HQQ, q_group_size=64)"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS / "model_size_quality.png",
    dpi=180,
)

plt.close()

# ============================================================
# FIGURE 2: INT4 QUALITY VS GROUP SIZE
# ============================================================

int4_quality = (
    group_quality[
        group_quality["cache"] == "int4"
    ]
    .sort_values("group_size")
)

plt.plot(
    int4_quality["group_size"],
    int4_quality[
        "retrieval_accuracy_pct"
    ],
    marker="o",
)

plt.xticks(
    [16, 32, 64, 128]
)

plt.ylim(
    -5,
    105,
)

plt.xlabel(
    "q_group_size"
)

plt.ylabel(
    "Retrieval exact-match, %"
)

plt.title(
    "Qwen3-1.7B: INT4 quality vs group size"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    PLOTS / "int4_group_size_quality.png",
    dpi=180,
)

plt.close()

# ============================================================
# FIGURE 3: QUALITY VS TOTAL GPU MEMORY SAVING
# ============================================================

for _, row in tradeoff.iterrows():

    plt.scatter(
        row["total_peak_saving_pct"],
        row["retrieval_accuracy_pct"],
    )

    label = row["cache"]

    if row["group_size"] != 0:
        label += (
            f" g{int(row['group_size'])}"
        )

    if row["cache"] != "int2":

        plt.annotate(
            label,
            (
                row["total_peak_saving_pct"],
                row["retrieval_accuracy_pct"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

plt.xlabel(
    "Total peak GPU memory saving vs Dynamic, %"
)

plt.ylabel(
    "Retrieval exact-match, %"
)

plt.title(
    "Qwen3-1.7B: quality-memory trade-off"
)

plt.ylim(
    -5,
    110,
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    PLOTS / "quality_memory_tradeoff.png",
    dpi=180,
)

plt.close()

performance17 = performance[
    performance["model_size"] == "1.7B"
].copy()

# ============================================================
# FIGURE 4: GPU MEMORY VS CONTEXT
# ============================================================

for cache in [
    "dynamic",
    "int4",
    "int2",
]:
    sub = (
        performance17[
            performance17["cache"]
            == cache
        ]
        .sort_values(
            "context_len"
        )
    )

    plt.plot(
        sub["context_len"],
        sub["peak_gpu_gib"],
        marker="o",
        label=cache,
    )


plt.xlabel(
    "Context length"
)

plt.ylabel(
    "Peak GPU memory, GiB"
)

plt.title(
    "Qwen3-1.7B: GPU memory vs context (q_group_size=64)"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS / "memory_vs_context_qwen17.png",
    dpi=180,
)

plt.close()

# ============================================================
# FIGURE 5: DECODE SPEED VS CONTEXT
# ============================================================

for cache in [
    "dynamic",
    "int4",
    "int2",
]:
    sub = (
        performance17[
            performance17["cache"]
            == cache
        ]
        .sort_values(
            "context_len"
        )
    )

    plt.plot(
        sub["context_len"],
        sub["decode_tok_s"],
        marker="o",
        label=cache,
    )


plt.xlabel(
    "Context length"
)

plt.ylabel(
    "Decode throughput, tokens/s"
)

plt.title(
    "Qwen3-1.7B: decode speed vs context (q_group_size=64)"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS / "decode_speed_vs_context_qwen17.png",
    dpi=180,
)

plt.close()

print()
print("Saved final plots:")
print(PLOTS / "model_size_quality.png")
print(PLOTS / "int4_group_size_quality.png")
print(PLOTS / "quality_memory_tradeoff.png")
print(PLOTS / "memory_vs_context_qwen17.png")
print(PLOTS / "decode_speed_vs_context_qwen17.png")
