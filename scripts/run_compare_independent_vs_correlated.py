"""
Compare independent vs correlated synthetic crowdsourcing experiments.

This script runs two synthetic experiments with the same benchmark pipeline:
    1. Independent annotators.
    2. Correlated annotators.

It then saves:
    - raw benchmark results,
    - aggregated summary tables,
    - delta tables: correlated minus independent,
    - comparison plots.

Required files at repo root:
    - generate_synthetic_data.py
    - generate_synthetic_data_correlated.py

Run from the repo root:
    python run_compare_independent_vs_correlated.py

Typical quick run:
    python run_compare_independent_vs_correlated.py --n-repet 3

Default run:
    python run_compare_independent_vs_correlated.py --n-repet 10
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from faircrowd.core import FairCrowdDataset
from faircrowd.metrics import (
    accuracy,
    f1_score,
    DemographicParity_,
    EqualOpportunities_,
    PredictiveParity_,
)
from faircrowd.competitors.fair_td import FairTD_Post
from faircrowd.competitors.geom_model import Majority_vote
from faircrowd.benchmark_new.comparison import evaluate_algorithms
from faircrowd.competitors import FairTD_ref, Optimal_reg, Geometric
from faircrowd.truth_inference.dawid_skene_multiple import DawidSkeneMultiple

from data_generators.generate_synthetic_data import generate_synhtetic_binary_restricted
from data_generators.generate_synthetic_data_correlated import (
    generate_synthetic_binary_correlated_restricted,
    empirical_error_correlation,
)


# ============================================================
# Dataset generation
# ============================================================

def make_faircrowd_dataset(
    Y_annotators: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
) -> FairCrowdDataset:
    """
    Build the FairCrowdDataset object expected by the repo.

    The x variable is a dummy zero column, as in the current synthetic scripts.
    """
    return FairCrowdDataset(
        pd.DataFrame(Y_annotators),
        pd.DataFrame(S[:, None]),
        pd.DataFrame(np.zeros(len(Y_annotators))),
        pd.DataFrame(Y[:, None]),
    )


def generate_independent_dataset(args) -> Tuple[FairCrowdDataset, Dict[str, float]]:
    """
    Generate the original independent-annotator synthetic dataset.
    """
    Y_annotators, S, Y, confusion, prop = generate_synhtetic_binary_restricted(
        N=args.N,
        R=args.R,
        R_anot_max=args.R_annots_max,
        bias_toward_1=args.bias_toward_1,
        prop_s_1=args.prop_s_1,
        rho_0=(args.rho_0_low, args.rho_0_high),
        rho_1=(args.rho_1_low, args.rho_1_high),
        seed=args.seed,
    )

    df = make_faircrowd_dataset(Y_annotators, S, Y)

    corr_errors = empirical_error_correlation(Y_annotators, Y)
    off_diag = corr_errors[~np.eye(corr_errors.shape[0], dtype=bool)]
    off_diag = off_diag[~np.isnan(off_diag)]

    diagnostics = {
        "prop_y_given_s0": float(prop[0]),
        "prop_y_given_s1": float(prop[1]),
        "mean_error_corr": float(np.mean(off_diag)) if len(off_diag) else np.nan,
        "median_error_corr": float(np.median(off_diag)) if len(off_diag) else np.nan,
        "min_error_corr": float(np.min(off_diag)) if len(off_diag) else np.nan,
        "max_error_corr": float(np.max(off_diag)) if len(off_diag) else np.nan,
    }

    return df, diagnostics


def generate_correlated_dataset(args) -> Tuple[FairCrowdDataset, Dict[str, float]]:
    """
    Generate the correlated-annotator synthetic dataset.
    """
    Y_annotators, S, Y, confusion, prop, details = generate_synthetic_binary_correlated_restricted(
        N=args.N,
        R=args.R,
        R_anot_max=args.R_annots_max,
        bias_toward_1=args.bias_toward_1,
        prop_s_1=args.prop_s_1,
        rho_0=(args.rho_0_low, args.rho_0_high),
        rho_1=(args.rho_1_low, args.rho_1_high),
        corr_0=args.corr_0,
        corr_1=args.corr_1,
        seed=args.seed,
        return_details=True,
    )

    df = make_faircrowd_dataset(Y_annotators, S, Y)

    corr_errors = empirical_error_correlation(Y_annotators, Y)
    off_diag = corr_errors[~np.eye(corr_errors.shape[0], dtype=bool)]
    off_diag = off_diag[~np.isnan(off_diag)]

    diagnostics = {
        "prop_y_given_s0": float(prop[0]),
        "prop_y_given_s1": float(prop[1]),
        "mean_error_corr": float(np.mean(off_diag)) if len(off_diag) else np.nan,
        "median_error_corr": float(np.median(off_diag)) if len(off_diag) else np.nan,
        "min_error_corr": float(np.min(off_diag)) if len(off_diag) else np.nan,
        "max_error_corr": float(np.max(off_diag)) if len(off_diag) else np.nan,
        "corr_0": float(args.corr_0),
        "corr_1": float(args.corr_1),
    }

    if "rho_by_group" in details:
        rho_by_group = details["rho_by_group"]
        diagnostics["mean_rho_s0"] = float(np.mean(rho_by_group[:, 0]))
        diagnostics["mean_rho_s1"] = float(np.mean(rho_by_group[:, 1]))

    return df, diagnostics


# ============================================================
# Algorithms
# ============================================================

def build_algorithms(df: FairCrowdDataset):
    """
    Build the same list of algorithms as in run_experiments_synhtetic.py.

    Important:
    The current repo script fits Dawid-Skene and Geometric once on the full
    synthetic dataset before evaluate_algorithms. This function keeps the same
    behavior, so that the independent and correlated cases are comparable with
    the current experimental protocol.
    """
    df_answers = df.answers.values
    df_s = df.s.values.flatten()
    df_y = df.y.values.flatten()

    n_classes = int(np.max(df["y"].values) + 1)
    n_annotators = len(df.answers.values[0])

    # Dawid-Skene
    dwm = DawidSkeneMultiple()
    _ = dwm.run(df)

    reg_DS = Optimal_reg(dwm)
    reg_DS.name = "FC_DS"

    post_td_DS = FairTD_Post(dwm)
    post_td_DS.name = "Post_TD_DS"

    # Geometric / Bayes
    geo = Geometric(n_classes, n_annotators)
    geo.fit(df_answers, df_s, df_y)

    reg_bayes = Optimal_reg(geo)
    reg_bayes.name = "FC_Bayes"

    post_td_bayes = FairTD_Post(geo)
    post_td_bayes.name = "Post_TD_Bayes"

    # Majority vote
    maj = Majority_vote(n_classes, n_annotators)

    reg_maj = Optimal_reg(maj)
    reg_maj.name = "FC_Maj"

    post_td_maj = FairTD_Post(maj)
    post_td_maj.name = "Post_TD_Maj"

    return [
        FairTD_ref(),
        post_td_bayes,
        reg_bayes,
        reg_maj,
        post_td_maj,
        reg_DS,
        post_td_DS,
    ]


# ============================================================
# Benchmark
# ============================================================

def run_one_scenario(
    scenario_name: str,
    df: FairCrowdDataset,
    args,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Run evaluate_algorithms on one scenario and add a scenario column.
    """
    print("\n" + "=" * 80)
    print(f"Running scenario: {scenario_name}")
    print("=" * 80)
    print("dataset size:", len(df.df))
    print("number annotators:", len(df.answers.values[0]))

    td_algorithms = build_algorithms(df)

    eps_list = args.eps_list
    accuracy_metrics = [accuracy, f1_score]
    fairness_metrics = [
        DemographicParity_(),
        EqualOpportunities_(),
        PredictiveParity_(),
    ]
    train_sizes_list = [args.train_size]

    result = evaluate_algorithms(
        df,
        td_algorithms,
        eps_list,
        accuracy_metrics,
        fairness_metrics,
        train_sizes_list=train_sizes_list,
        n_repet=args.n_repet,
        filename_prefix=f"{scenario_name}_synthetic",
        save_dir=str(output_dir / "repo_eval_csv"),
        full_test=False,
    )

    result = result.copy()
    result.insert(0, "scenario", scenario_name)

    return result


# ============================================================
# Tables
# ============================================================

def build_summary_tables(results: pd.DataFrame, output_dir: Path):
    """
    Save useful summary tables.
    """
    metric_cols = [
        col
        for col in [
            "accuracy",
            "f1_score",
            "DemographicParity_",
            "EqualOpportunities_",
            "PredictiveParity_",
            "execution_time",
        ]
        if col in results.columns
    ]

    grouped = (
        results
        .groupby(["scenario", "algorithm", "epsilon", "train_size"], as_index=False)
        .agg({col: ["mean", "std"] for col in metric_cols})
    )

    grouped.columns = [
        "_".join([c for c in col if c])
        if isinstance(col, tuple)
        else col
        for col in grouped.columns
    ]

    summary_path = output_dir / "summary_mean_std_by_scenario_algorithm_epsilon.csv"
    grouped.to_csv(summary_path, index=False)

    # Pivot into independent vs correlated on the same algorithm/epsilon.
    mean_cols = [c for c in grouped.columns if c.endswith("_mean")]
    pivot = grouped.pivot_table(
        index=["algorithm", "epsilon", "train_size"],
        columns="scenario",
        values=mean_cols,
    )

    # Flatten columns: e.g. f1_score_mean__correlated
    pivot.columns = [f"{metric}__{scenario}" for metric, scenario in pivot.columns]
    pivot = pivot.reset_index()

    for metric in mean_cols:
        independent_col = f"{metric}__independent"
        correlated_col = f"{metric}__correlated"
        if independent_col in pivot.columns and correlated_col in pivot.columns:
            pivot[f"delta_{metric}_correlated_minus_independent"] = (
                pivot[correlated_col] - pivot[independent_col]
            )

    delta_path = output_dir / "delta_correlated_minus_independent.csv"
    pivot.to_csv(delta_path, index=False)

    # Best per scenario/algorithm according to F1, and then according to DP.
    if "f1_score" in results.columns:
        mean_for_best = (
            results
            .groupby(["scenario", "algorithm", "epsilon"], as_index=False)
            .agg(
                f1_score_mean=("f1_score", "mean"),
                dp_mean=("DemographicParity_", "mean"),
                eo_mean=("EqualOpportunities_", "mean"),
                pp_mean=("PredictiveParity_", "mean"),
            )
        )

        best_f1 = (
            mean_for_best
            .sort_values(["scenario", "algorithm", "f1_score_mean"], ascending=[True, True, False])
            .groupby(["scenario", "algorithm"], as_index=False)
            .head(1)
        )

        best_f1.to_csv(output_dir / "best_epsilon_by_f1.csv", index=False)

        best_dp = (
            mean_for_best
            .sort_values(["scenario", "algorithm", "dp_mean"], ascending=[True, True, True])
            .groupby(["scenario", "algorithm"], as_index=False)
            .head(1)
        )

        best_dp.to_csv(output_dir / "best_epsilon_by_demographic_parity.csv", index=False)

    return grouped, pivot


# ============================================================
# Plots
# ============================================================

def _aggregate_for_plots(
    results: pd.DataFrame,
    dp_col: str = "DemographicParity_",
    f1_col: str = "f1_score",
) -> pd.DataFrame:
    return (
        results
        .groupby(["scenario", "algorithm", "epsilon"], as_index=False)
        .agg(
            dp_mean=(dp_col, "mean"),
            dp_std=(dp_col, "std"),
            f1_mean=(f1_col, "mean"),
            f1_std=(f1_col, "std"),
        )
    )


def plot_tradeoff_by_scenario(
    results: pd.DataFrame,
    output_dir: Path,
):
    """
    Save one F1 vs Demographic Parity plot per scenario.
    """
    agg = _aggregate_for_plots(results)

    for scenario, scenario_df in agg.groupby("scenario"):
        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        for algorithm, subdf in scenario_df.groupby("algorithm"):
            subdf = subdf.sort_values("epsilon")

            ax.plot(
                subdf["dp_mean"].values,
                subdf["f1_mean"].values,
                marker="o",
                linewidth=2,
                label=algorithm,
            )

            ax.fill_between(
                subdf["dp_mean"].values,
                subdf["f1_mean"].values - subdf["f1_std"].fillna(0).values,
                subdf["f1_mean"].values + subdf["f1_std"].fillna(0).values,
                alpha=0.2,
            )

            for _, row in subdf.iterrows():
                ax.annotate(
                    f"{row['epsilon']:.2f}",
                    (row["dp_mean"], row["f1_mean"]),
                    fontsize=8,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

        ax.set_xlabel("Demographic Parity gap")
        ax.set_ylabel("F1 score")
        ax.set_title(f"F1 vs Demographic Parity — {scenario}")
        ax.legend(
            title="Algorithm",
            fontsize=8,
            title_fontsize=9,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout(rect=[0, 0, 0.78, 1])

        for ext in ["png", "pdf"]:
            fig.savefig(output_dir / f"tradeoff_{scenario}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_overlay_independent_vs_correlated(
    results: pd.DataFrame,
    output_dir: Path,
):
    """
    Overlay independent and correlated tradeoff curves.

    Labels are algorithm + scenario. This is intentionally dense but useful.
    """
    agg = _aggregate_for_plots(results)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    linestyles = {
        "independent": "-",
        "correlated": "--",
    }

    markers = {
        "independent": "o",
        "correlated": "s",
    }

    for (algorithm, scenario), subdf in agg.groupby(["algorithm", "scenario"]):
        subdf = subdf.sort_values("epsilon")

        ax.plot(
            subdf["dp_mean"].values,
            subdf["f1_mean"].values,
            linestyle=linestyles.get(scenario, "-"),
            marker=markers.get(scenario, "o"),
            linewidth=2,
            label=f"{algorithm} | {scenario}",
        )

    ax.set_xlabel("Demographic Parity gap")
    ax.set_ylabel("F1 score")
    ax.set_title("Independent vs correlated annotators")
    ax.legend(
        title="Algorithm | scenario",
        fontsize=7,
        title_fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 0.72, 1])

    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"overlay_tradeoff_independent_vs_correlated.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_curves(
    delta_table: pd.DataFrame,
    output_dir: Path,
):
    """
    Plot delta F1 and delta Demographic Parity:
        correlated mean - independent mean.
    """
    delta_f1_col = "delta_f1_score_mean_correlated_minus_independent"
    delta_dp_col = "delta_DemographicParity__mean_correlated_minus_independent"

    if delta_f1_col in delta_table.columns:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        for algorithm, subdf in delta_table.groupby("algorithm"):
            subdf = subdf.sort_values("epsilon")
            ax.plot(
                subdf["epsilon"].values,
                subdf[delta_f1_col].values,
                marker="o",
                linewidth=2,
                label=algorithm,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("Delta F1: correlated - independent")
        ax.set_title("Effect of annotator correlation on F1")
        ax.legend(
            title="Algorithm",
            fontsize=8,
            title_fontsize=9,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout(rect=[0, 0, 0.78, 1])

        for ext in ["png", "pdf"]:
            fig.savefig(output_dir / f"delta_f1_correlated_minus_independent.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if delta_dp_col in delta_table.columns:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        for algorithm, subdf in delta_table.groupby("algorithm"):
            subdf = subdf.sort_values("epsilon")
            ax.plot(
                subdf["epsilon"].values,
                subdf[delta_dp_col].values,
                marker="o",
                linewidth=2,
                label=algorithm,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("Delta DP gap: correlated - independent")
        ax.set_title("Effect of annotator correlation on Demographic Parity")
        ax.legend(
            title="Algorithm",
            fontsize=8,
            title_fontsize=9,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout(rect=[0, 0, 0.78, 1])

        for ext in ["png", "pdf"]:
            fig.savefig(output_dir / f"delta_dp_correlated_minus_independent.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# Diagnostics
# ============================================================

def save_diagnostics(
    diagnostics: Dict[str, Dict[str, float]],
    output_dir: Path,
):
    rows = []
    for scenario, d in diagnostics.items():
        row = {"scenario": scenario}
        row.update(d)
        rows.append(row)

    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(output_dir / "dataset_diagnostics.csv", index=False)

    print("\nDataset diagnostics:")
    print(diag_df.to_string(index=False))


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--R-annots-max", type=int, default=5, dest="R_annots_max")

    parser.add_argument("--bias-toward-1", type=float, default=0.2, dest="bias_toward_1")
    parser.add_argument("--prop-s-1", type=float, default=0.6, dest="prop_s_1")

    parser.add_argument("--rho-0-low", type=float, default=0.0)
    parser.add_argument("--rho-0-high", type=float, default=0.5)
    parser.add_argument("--rho-1-low", type=float, default=0.0)
    parser.add_argument("--rho-1-high", type=float, default=0.4)

    parser.add_argument("--corr-0", type=float, default=0.5)
    parser.add_argument("--corr-1", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-repet", type=int, default=10)
    parser.add_argument("--train-size", type=float, default=0.4)

    parser.add_argument(
        "--eps-list",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.2],
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures_synthetic_compare_corr",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "repo_eval_csv").mkdir(parents=True, exist_ok=True)

    print("Output directory:", output_dir.resolve())

    # Generate both datasets
    df_independent, diag_independent = generate_independent_dataset(args)
    df_correlated, diag_correlated = generate_correlated_dataset(args)

    save_diagnostics(
        {
            "independent": diag_independent,
            "correlated": diag_correlated,
        },
        output_dir,
    )

    # Run benchmarks
    results_independent = run_one_scenario("independent", df_independent, args, output_dir)
    results_correlated = run_one_scenario("correlated", df_correlated, args, output_dir)

    results = pd.concat([results_independent, results_correlated], ignore_index=True)

    raw_path = output_dir / "raw_results_independent_vs_correlated.csv"
    results.to_csv(raw_path, index=False)

    print("\nRaw results saved to:")
    print(raw_path)

    summary, delta_table = build_summary_tables(results, output_dir)

    print("\nSummary table saved to:")
    print(output_dir / "summary_mean_std_by_scenario_algorithm_epsilon.csv")

    print("\nDelta table saved to:")
    print(output_dir / "delta_correlated_minus_independent.csv")

    # Plots
    plot_tradeoff_by_scenario(results, output_dir)
    plot_overlay_independent_vs_correlated(results, output_dir)
    plot_delta_curves(delta_table, output_dir)

    print("\nPlots saved in:")
    print(output_dir)

    # Small terminal preview
    preview_cols = [
        "algorithm",
        "epsilon",
        "train_size",
        "delta_f1_score_mean_correlated_minus_independent",
        "delta_DemographicParity__mean_correlated_minus_independent",
    ]
    existing_preview_cols = [c for c in preview_cols if c in delta_table.columns]

    print("\nDelta preview:")
    print(delta_table[existing_preview_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
