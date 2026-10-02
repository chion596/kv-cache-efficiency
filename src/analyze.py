import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_line_plot(
    summary,
    value_col,
    std_col,
    ylabel,
    title,
    output_path,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for cache in sorted(summary["cache"].unique()):
        part = (
            summary[summary["cache"] == cache]
            .sort_values("context_len")
        )

        ax.errorbar(
            part["context_len"],
            part[value_col],
            yerr=part[std_col],
            marker="o",
            capsize=3,
            label=cache,
        )

    contexts = sorted(summary["context_len"].unique())

    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([str(x) for x in contexts])

    ax.set_xlabel("Длина контекста, токены")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="CSV полного benchmark",
    )

    parser.add_argument(
        "--summary-dir",
        default="results/summary/baseline",
    )

    parser.add_argument(
        "--plots-dir",
        default="plots/baseline",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    summary_dir = Path(args.summary_dir)
    plots_dir = Path(args.plots_dir)

    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    df["error"] = (
        df["error"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    failed = df[df["error"] != ""]
    valid = df[df["error"] == ""].copy()

    print("=" * 70)
    print("BASELINE ANALYSIS")
    print("=" * 70)
    print("Input:", input_path)
    print("Всего строк:", len(df))
    print("Успешных строк:", len(valid))
    print("Строк с ошибками:", len(failed))

    if len(valid) == 0:
        raise RuntimeError("Нет успешных измерений")

    summary = (
        valid
        .groupby(
            ["context_len", "cache"],
            as_index=False,
        )
        .agg(
            repeats=("repeat", "count"),

            prefill_sec_mean=("prefill_sec", "mean"),
            prefill_sec_std=("prefill_sec", "std"),

            ttft_ms_mean=("ttft_ms", "mean"),
            ttft_ms_std=("ttft_ms", "std"),

            decode_tok_s_mean=("decode_tok_s", "mean"),
            decode_tok_s_std=("decode_tok_s", "std"),

            tpot_ms_mean=("tpot_ms", "mean"),
            tpot_ms_std=("tpot_ms", "std"),

            e2e_tok_s_mean=("e2e_tok_s", "mean"),
            e2e_tok_s_std=("e2e_tok_s", "std"),

            peak_allocated_gib_mean=(
                "total_peak_allocated_gib",
                "mean",
            ),
            peak_allocated_gib_std=(
                "total_peak_allocated_gib",
                "std",
            ),

            theoretical_kv_gib=(
                "theoretical_kv_prompt_gib",
                "mean",
            ),

            physical_kv_gpu_gib=(
                "cache_prompt_gpu_gib",
                "mean",
            ),

            physical_kv_cpu_gib=(
                "cache_prompt_cpu_gib",
                "mean",
            ),

            output_hash_nunique=(
                "output_hash",
                "nunique",
            ),
        )
        .sort_values(["context_len", "cache"])
    )

    for col in summary.columns:
        if col.endswith("_std"):
            summary[col] = summary[col].fillna(0.0)

    summary_path = (
        summary_dir
        / "full_dynamic_static_qwen3_0.6b_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ----------------------------------------------------------
    # Direct Dynamic vs Static comparison
    # ----------------------------------------------------------

    dynamic = (
        summary[summary["cache"] == "dynamic"]
        .set_index("context_len")
    )

    static = (
        summary[summary["cache"] == "static"]
        .set_index("context_len")
    )

    common_contexts = dynamic.index.intersection(static.index)

    comparison = pd.DataFrame(
        index=common_contexts
    )

    comparison.index.name = "context_len"

    comparison["dynamic_decode_tok_s"] = (
        dynamic.loc[
            common_contexts,
            "decode_tok_s_mean",
        ]
    )

    comparison["static_decode_tok_s"] = (
        static.loc[
            common_contexts,
            "decode_tok_s_mean",
        ]
    )

    comparison["static_decode_slowdown_pct"] = (
        100
        * (
            1
            - comparison["static_decode_tok_s"]
            / comparison["dynamic_decode_tok_s"]
        )
    )

    comparison["dynamic_tpot_ms"] = (
        dynamic.loc[
            common_contexts,
            "tpot_ms_mean",
        ]
    )

    comparison["static_tpot_ms"] = (
        static.loc[
            common_contexts,
            "tpot_ms_mean",
        ]
    )

    comparison["static_tpot_penalty_pct"] = (
        100
        * (
            comparison["static_tpot_ms"]
            / comparison["dynamic_tpot_ms"]
            - 1
        )
    )

    comparison["dynamic_peak_gib"] = (
        dynamic.loc[
            common_contexts,
            "peak_allocated_gib_mean",
        ]
    )

    comparison["static_peak_gib"] = (
        static.loc[
            common_contexts,
            "peak_allocated_gib_mean",
        ]
    )

    comparison["static_peak_extra_mib"] = (
        (
            comparison["static_peak_gib"]
            - comparison["dynamic_peak_gib"]
        )
        * 1024
    )

    comparison_path = (
        summary_dir
        / "dynamic_static_comparison_qwen3_0.6b.csv"
    )

    comparison.reset_index().to_csv(
        comparison_path,
        index=False,
    )

    # ----------------------------------------------------------
    # Plots
    # ----------------------------------------------------------

    save_line_plot(
        summary,
        value_col="decode_tok_s_mean",
        std_col="decode_tok_s_std",
        ylabel="Decode throughput, токенов/с",
        title="Скорость decode в зависимости от длины контекста",
        output_path=(
            plots_dir
            / "decode_throughput_vs_context.png"
        ),
    )

    save_line_plot(
        summary,
        value_col="tpot_ms_mean",
        std_col="tpot_ms_std",
        ylabel="TPOT, мс",
        title="Time Per Output Token",
        output_path=(
            plots_dir
            / "tpot_vs_context.png"
        ),
    )

    save_line_plot(
        summary,
        value_col="prefill_sec_mean",
        std_col="prefill_sec_std",
        ylabel="Время prefill, с",
        title="Prefill в зависимости от длины контекста",
        output_path=(
            plots_dir
            / "prefill_time_vs_context.png"
        ),
    )

    save_line_plot(
        summary,
        value_col="peak_allocated_gib_mean",
        std_col="peak_allocated_gib_std",
        ylabel="Peak allocated GPU memory, GiB",
        title="GPU-память в зависимости от длины контекста",
        output_path=(
            plots_dir
            / "gpu_memory_vs_context.png"
        ),
    )

    # ----------------------------------------------------------
    # Theoretical vs physical KV
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    theoretical = (
        summary
        .groupby("context_len", as_index=False)
        ["theoretical_kv_gib"]
        .mean()
        .sort_values("context_len")
    )

    ax.plot(
        theoretical["context_len"],
        theoretical["theoretical_kv_gib"],
        marker="o",
        linestyle="--",
        label="Теоретический FP16 KV Cache",
    )

    for cache in ["dynamic", "static"]:
        part = (
            summary[summary["cache"] == cache]
            .sort_values("context_len")
        )

        ax.plot(
            part["context_len"],
            part["physical_kv_gpu_gib"],
            marker="o",
            label=f"{cache}: фактически выделено",
        )

    contexts = sorted(summary["context_len"].unique())

    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([str(x) for x in contexts])

    ax.set_xlabel("Длина контекста, токены")
    ax.set_ylabel("Размер KV Cache, GiB")
    ax.set_title(
        "Теоретический и фактический размер KV Cache"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        plots_dir / "theoretical_vs_physical_kv.png",
        dpi=180,
    )
    plt.close(fig)

    # ----------------------------------------------------------
    # Console report
    # ----------------------------------------------------------

    print()
    print("DynamicCache vs StaticCache")
    print("-" * 70)

    print(
        comparison[
            [
                "dynamic_decode_tok_s",
                "static_decode_tok_s",
                "static_decode_slowdown_pct",
                "static_peak_extra_mib",
            ]
        ]
        .round(3)
        .to_string()
    )

    print()
    print("Проверка output hashes:")
    print(
        summary[
            [
                "context_len",
                "cache",
                "output_hash_nunique",
            ]
        ].to_string(index=False)
    )

    print()
    print("Summary:", summary_path)
    print("Comparison:", comparison_path)
    print("Plots:", plots_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
