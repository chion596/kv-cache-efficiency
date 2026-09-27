import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument(
        "--summary-dir",
        default="results/summary/static_compile",
    )
    p.add_argument(
        "--plots-dir",
        default="plots/static_compile",
    )
    args = p.parse_args()

    summary_dir = Path(args.summary_dir)
    plots_dir = Path(args.plots_dir)

    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    errors = (
        df["error"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    m = df[
        (df["phase"] == "measure")
        & (errors == "")
    ].copy()

    summary = (
        m.groupby(["context_len", "mode"], as_index=False)
        .agg(
            repeats=("repeat", "count"),
            elapsed_sec_mean=("elapsed_sec", "mean"),
            elapsed_sec_std=("elapsed_sec", "std"),
            e2e_tok_s_mean=("e2e_tok_s", "mean"),
            e2e_tok_s_std=("e2e_tok_s", "std"),
            peak_allocated_gib_mean=(
                "peak_allocated_gib", "mean"
            ),
            output_hash_nunique=(
                "output_hash", "nunique"
            ),
            all_correct=(
                "same_as_dynamic", "all"
            ),
        )
        .sort_values(["context_len", "mode"])
    )

    summary.to_csv(
        summary_dir / "static_compile_summary.csv",
        index=False,
    )

    pivot = (
        summary
        .pivot(
            index="context_len",
            columns="mode",
            values="elapsed_sec_mean",
        )
        .sort_index()
    )

    comparison = pd.DataFrame(
        index=pivot.index
    )

    comparison["dynamic_eager_sec"] = (
        pivot["dynamic_eager"]
    )

    comparison["static_eager_sec"] = (
        pivot["static_eager"]
    )

    comparison["static_compiled_sec"] = (
        pivot["static_compiled"]
    )

    comparison["compile_speedup_vs_static_x"] = (
        comparison["static_eager_sec"]
        / comparison["static_compiled_sec"]
    )

    comparison[
        "compile_reduction_vs_static_pct"
    ] = (
        100
        * (
            1
            - comparison["static_compiled_sec"]
            / comparison["static_eager_sec"]
        )
    )

    comparison[
        "compiled_vs_dynamic_pct"
    ] = (
        100
        * (
            comparison["static_compiled_sec"]
            / comparison["dynamic_eager_sec"]
            - 1
        )
    )

    comparison.reset_index().to_csv(
        summary_dir
        / "static_compile_comparison.csv",
        index=False,
    )

    # E2E latency
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode in [
        "dynamic_eager",
        "static_eager",
        "static_compiled",
    ]:
        part = summary[
            summary["mode"] == mode
        ].sort_values("context_len")

        ax.errorbar(
            part["context_len"],
            part["elapsed_sec_mean"],
            yerr=part["elapsed_sec_std"],
            marker="o",
            capsize=3,
            label=mode,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Длина контекста, токены")
    ax.set_ylabel("End-to-end latency, с")
    ax.set_title(
        "Влияние torch.compile на StaticCache"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        plots_dir
        / "static_compile_latency.png",
        dpi=180,
    )

    plt.close(fig)

    # Compile benefit vs Static eager
    fig, ax = plt.subplots(figsize=(8, 5))

    c = comparison.reset_index()

    ax.plot(
        c["context_len"],
        c["compile_reduction_vs_static_pct"],
        marker="o",
    )

    ax.axhline(0, linewidth=1)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Длина контекста, токены")
    ax.set_ylabel(
        "Снижение latency относительно Static eager, %"
    )
    ax.set_title(
        "Практический выигрыш torch.compile"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        plots_dir
        / "compile_benefit_vs_context.png",
        dpi=180,
    )

    plt.close(fig)

    # Compiled Static vs Dynamic
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        c["context_len"],
        c["compiled_vs_dynamic_pct"],
        marker="o",
    )

    ax.axhline(0, linewidth=1)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Длина контекста, токены")
    ax.set_ylabel(
        "Static compiled относительно Dynamic, % latency"
    )
    ax.set_title(
        "Crossover Static compiled и Dynamic"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        plots_dir
        / "compiled_vs_dynamic.png",
        dpi=180,
    )

    plt.close(fig)

    print(summary.round(4).to_string(index=False))

    print()
    print("Comparison:")
    print(
        comparison.round(4).to_string()
    )


if __name__ == "__main__":
    main()
