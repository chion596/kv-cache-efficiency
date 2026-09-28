from pathlib import Path
import pandas as pd


FILES = {
    "0.6B": "results/raw/quality_hqq_qwen3_0.6b_4355340.csv",
    "1.7B": "results/raw/quality_hqq_qwen3_1.7b_4358632.csv",
    "4B": "results/raw/quality_hqq_qwen3_4b_4358633.csv",
    "8B": "results/raw/quality_hqq_qwen3_8b_4357836.csv",
}

OUT = Path("results/summary/quality_scaling")
OUT.mkdir(parents=True, exist_ok=True)

rows = []

for model_size, path in FILES.items():
    df = pd.read_csv(path)

    errors = (
        df["error"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[errors == ""].copy()

    def first_token_correct(row):
        n = int(row["target_token_count"])

        all_correct = round(
            row["tf_top1_all_pct"] / 100 * n
        )

        decode_correct = round(
            row["tf_top1_decode_pct"] / 100 * (n - 1)
        )

        return int(
            all_correct - decode_correct > 0
        )

    df["first_token_correct"] = df.apply(
        first_token_correct,
        axis=1,
    )

    for cache, group in df.groupby("cache"):
        rows.append({
            "model_size": model_size,
            "cache": cache,
            "n": len(group),

            "retrieval_accuracy_pct":
                100 * group[
                    "normalized_exact_match"
                ].mean(),

            "first_token_accuracy_pct":
                100 * group[
                    "first_token_correct"
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

summary = pd.DataFrame(rows)

order = {
    "0.6B": 0,
    "1.7B": 1,
    "4B": 2,
    "8B": 3,
}

summary["_order"] = summary[
    "model_size"
].map(order)

summary = (
    summary.sort_values(
        ["_order", "cache"]
    )
    .drop(columns="_order")
)

summary.to_csv(
    OUT / "quality_scaling.csv",
    index=False,
)

print("=" * 100)
print("QUALITY SCALING")
print("=" * 100)
print(
    summary.round(4).to_string(
        index=False
    )
)
