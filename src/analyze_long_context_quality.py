from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = (
    "results/raw/"
    "quality_hqq_qwen3_0.6b_4355340.csv"
)

OUT = Path(
    "results/summary/quality_hqq"
)

PLOTS = Path(
    "plots/quality_hqq"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PLOTS.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(INPUT)

errors = (
    df["error"]
    .fillna("")
    .astype(str)
    .str.strip()
)

good = df[
    errors == ""
].copy()


print("=" * 100)
print("GENERAL")
print("=" * 100)

print("all rows:", len(df))
print("good rows:", len(good))
print(
    "errors:",
    int((errors != "").sum()),
)

print()
print(
    good.groupby(
        [
            "cache",
            "context_len",
            "position",
        ]
    )
    .size()
    .to_string()
)


# ============================================================
# OVERALL
# ============================================================

overall = (
    good.groupby(
        "cache",
        as_index=False,
    )
    .agg(
        n=(
            "seed",
            "count",
        ),

        retrieval_accuracy=(
            "normalized_exact_match",
            "mean",
        ),

        strict_accuracy=(
            "strict_exact_match",
            "mean",
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
)

overall[
    "retrieval_accuracy"
] *= 100

overall[
    "strict_accuracy"
] *= 100

overall.to_csv(
    OUT / "overall.csv",
    index=False,
)


# ============================================================
# CONTEXT
# ============================================================

by_context = (
    good.groupby(
        [
            "cache",
            "context_len",
        ],
        as_index=False,
    )
    .agg(
        n=(
            "seed",
            "count",
        ),

        retrieval_accuracy=(
            "normalized_exact_match",
            "mean",
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
)

by_context[
    "retrieval_accuracy"
] *= 100

by_context.to_csv(
    OUT / "by_context.csv",
    index=False,
)


# ============================================================
# POSITION
# ============================================================

by_position = (
    good.groupby(
        [
            "cache",
            "position",
        ],
        as_index=False,
    )
    .agg(
        n=(
            "seed",
            "count",
        ),

        retrieval_accuracy=(
            "normalized_exact_match",
            "mean",
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
)

by_position[
    "retrieval_accuracy"
] *= 100

by_position.to_csv(
    OUT / "by_position.csv",
    index=False,
)


# ============================================================
# CONTEXT × POSITION
# ============================================================

grid = (
    good.groupby(
        [
            "cache",
            "context_len",
            "position",
        ],
        as_index=False,
    )
    .agg(
        n=(
            "seed",
            "count",
        ),

        correct=(
            "normalized_exact_match",
            "sum",
        ),

        retrieval_accuracy=(
            "normalized_exact_match",
            "mean",
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
)

grid[
    "retrieval_accuracy"
] *= 100

grid.to_csv(
    OUT / "context_position_grid.csv",
    index=False,
)


print()
print("=" * 100)
print("OVERALL")
print("=" * 100)

print(
    overall
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 100)
print("BY CONTEXT")
print("=" * 100)

print(
    by_context
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 100)
print("BY POSITION")
print("=" * 100)

print(
    by_position
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 100)
print("CONTEXT × POSITION")
print("=" * 100)

print(
    grid
    .round(4)
    .to_string(index=False)
)


# ============================================================
# PLOTS
# ============================================================

for cache in [
    "dynamic",
    "int4",
    "int2",
]:
    s = (
        by_context[
            by_context.cache
            == cache
        ]
        .sort_values(
            "context_len"
        )
    )

    plt.plot(
        s.context_len,
        s.retrieval_accuracy,
        marker="o",
        label=cache,
    )

plt.xlabel(
    "Context length"
)

plt.ylabel(
    "Retrieval exact-match, %"
)

plt.title(
    "Long-context retrieval quality"
)

plt.ylim(
    -5,
    105,
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS
    / "retrieval_accuracy_vs_context.png",
    dpi=180,
)

plt.close()


for cache in [
    "dynamic",
    "int4",
    "int2",
]:
    s = (
        by_context[
            by_context.cache
            == cache
        ]
        .sort_values(
            "context_len"
        )
    )

    plt.plot(
        s.context_len,
        s.tf_decode_nll,
        marker="o",
        label=cache,
    )

plt.xlabel(
    "Context length"
)

plt.ylabel(
    "Teacher-forced decode NLL"
)

plt.title(
    "Probability degradation after KV quantization"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS
    / "teacher_forced_nll_vs_context.png",
    dpi=180,
)

plt.close()


print()
print("Saved summaries:", OUT)
print("Saved plots:", PLOTS)
