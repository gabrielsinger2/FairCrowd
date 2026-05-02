"""
Run FC post-processing experiments on the Measuring Hate Speech dataset.

Dataset:
    ucberkeley-dlab/measuring-hate-speech

Key fixes vs previous version:
  - choose_s_col_auto now selects the attribute with the LARGEST gold DP gap
    (subject to minimum group size), so the fairness benchmark actually has
    a bias to correct.
  - Default y_threshold is 0.5 (the official binary cutoff per the dataset card).
  - BROAD targets (target_race, target_gender, ...) are INCLUDED by default,
    because these have enough samples AND meaningful DP gaps.
  - Stricter min_annotations_per_item default (5 instead of 2).
  - Float annotations explicitly clipped to {0,1} to avoid casting edge cases.

Usage:
    python run_experiments_measuring_hate_speech.py --diagnose-only
    python run_experiments_measuring_hate_speech.py --max-items 6000 --max-annotators 800 --n-repet 10
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from faircrowd.benchmark_new.comparison import (
    evaluate_algorithms,
    plot_f1_vs_demographic_parity_with_variance,
)
from faircrowd.competitors import FairTD_ref, Optimal_reg, Geometric
from faircrowd.truth_inference.dawid_skene_multiple import DawidSkeneMultiple


def load_raw_measuring_hate_speech() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Run: pip install datasets pyarrow"
        ) from exc

    try:
        ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", "binary")
    except Exception:
        ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")

    if "train" not in ds:
        raise RuntimeError("No 'train' split in the HF dataset.")
    return ds["train"].to_pandas()


BROAD_TARGET_COLUMNS = {
    "target_race",
    "target_religion",
    "target_origin",
    "target_gender",
    "target_sexuality",
    "target_age",
    "target_politics",
}


def get_target_columns(raw: pd.DataFrame, only_broad: bool = True) -> List[str]:
    cols = [c for c in raw.columns if c.startswith("target_")]
    if only_broad:
        cols = [c for c in cols if c in BROAD_TARGET_COLUMNS]
    return sorted(cols)


def item_level_table(raw: pd.DataFrame) -> pd.DataFrame:
    required = ["comment_id", "hate_speech_score", "text"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return raw.sort_values("comment_id").groupby("comment_id", as_index=True).first()


def build_target_imbalance_report(
    raw: pd.DataFrame,
    y_threshold: float = 0.5,
    only_broad: bool = True,
) -> pd.DataFrame:
    items = item_level_table(raw).copy()
    items["Y"] = (items["hate_speech_score"] > y_threshold).astype(int)

    rows = []
    for col in get_target_columns(raw, only_broad=only_broad):
        if col not in items.columns:
            continue

        s = items[col].fillna(False).astype(bool).astype(int)
        n = len(s)
        n_s1 = int(s.sum())
        n_s0 = int(n - n_s1)
        if n_s1 == 0 or n_s0 == 0:
            continue

        y = items["Y"].values
        p_y1_s1 = float(y[s.values == 1].mean())
        p_y1_s0 = float(y[s.values == 0].mean())

        rows.append({
            "s_col": col,
            "n_items": n,
            "n_s1": n_s1,
            "n_s0": n_s0,
            "p_s1": n_s1 / n,
            "p_y1": float(y.mean()),
            "p_y1_given_s1": p_y1_s1,
            "p_y1_given_s0": p_y1_s0,
            "abs_dp_gold": abs(p_y1_s1 - p_y1_s0),
        })

    report = pd.DataFrame(rows)
    if len(report) == 0:
        return report

    # Sort by DP gap DESCENDING — we want strong fairness signal.
    report = report.sort_values("abs_dp_gold", ascending=False).reset_index(drop=True)
    return report


def choose_s_col_auto(
    report: pd.DataFrame,
    min_s1_items: int = 300,
    min_s1_rate: float = 0.05,
    max_s1_rate: float = 0.60,
    min_dp_gold: float = 0.10,
) -> str:
    """
    Pick the attribute with the LARGEST DP gold gap, subject to enough samples
    in the rare group. This is the inverse of the previous strategy.
    """
    if len(report) == 0:
        raise ValueError("Empty target imbalance report.")

    candidates = report[
        (report["n_s1"] >= min_s1_items)
        & (report["p_s1"] >= min_s1_rate)
        & (report["p_s1"] <= max_s1_rate)
        & (report["abs_dp_gold"] >= min_dp_gold)
    ].copy()

    if len(candidates) == 0:
        # Relax: keep only sample-size constraints, but still rank by DP.
        candidates = report[report["n_s1"] >= min_s1_items].copy()
    if len(candidates) == 0:
        candidates = report.copy()

    chosen = candidates.sort_values("abs_dp_gold", ascending=False).iloc[0]
    return str(chosen["s_col"])


# ============================================================
# Annotation conversion
# ============================================================

def make_binary_annotation(raw: pd.DataFrame, annotation_col: str, threshold: float) -> pd.Series:
    if annotation_col not in raw.columns:
        raise ValueError(f"annotation_col={annotation_col!r} missing.")
    values = pd.to_numeric(raw[annotation_col], errors="coerce")
    return (values > threshold).astype(float)


def iterative_filter_answers(
    answers: pd.DataFrame,
    min_annotations_per_item: int,
    min_annotations_per_annotator: int,
    max_iter: int = 10,
) -> pd.DataFrame:
    current = answers.copy()
    for _ in range(max_iter):
        old_shape = current.shape
        item_counts = current.notna().sum(axis=1)
        current = current.loc[item_counts >= min_annotations_per_item]
        annot_counts = current.notna().sum(axis=0)
        current = current.loc[:, annot_counts >= min_annotations_per_annotator]
        if current.shape == old_shape:
            break
    return current


def build_mhs_faircrowd_dataset(
    raw: pd.DataFrame,
    s_col: str = "auto",
    y_threshold: float = 0.5,
    annotation_col: str = "hatespeech",
    annotation_threshold: float = 0.5,
    max_items: int = 5000,
    max_annotators: int = 800,
    min_annotations_per_item: int = 5,
    min_annotations_per_annotator: int = 5,
    random_state: int = 0,
    only_broad: bool = True,
    save_processed_dir: Optional[Path] = None,
) -> Tuple[FairCrowdDataset, Dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    required = ["comment_id", "annotator_id", "hate_speech_score", "text", annotation_col]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    report = build_target_imbalance_report(raw, y_threshold=y_threshold, only_broad=only_broad)

    if s_col == "auto":
        s_col = choose_s_col_auto(report)
        print(f"[auto] selected sensitive target column: {s_col} "
              f"(abs_dp_gold = {report.loc[report['s_col']==s_col, 'abs_dp_gold'].values[0]:.3f})")

    if s_col not in raw.columns:
        raise ValueError(f"s_col={s_col!r} not found.")

    working = raw.copy()
    working["annot_label"] = make_binary_annotation(
        working, annotation_col=annotation_col, threshold=annotation_threshold
    )

    items = item_level_table(working).copy()
    items["S"] = items[s_col].fillna(False).astype(bool).astype(int)
    items["Y"] = (pd.to_numeric(items["hate_speech_score"], errors="coerce") > y_threshold).astype(int)
    items["text_len"] = items["text"].fillna("").astype(str).str.len()

    working = working.dropna(subset=["comment_id", "annotator_id", "annot_label"])
    working = working[working["comment_id"].isin(items.index)]

    annot_counts_global = working.groupby("annotator_id").size().sort_values(ascending=False)
    if max_annotators is not None and max_annotators > 0:
        kept_annotators = annot_counts_global.head(max_annotators).index
        working = working[working["annotator_id"].isin(kept_annotators)]

    counts_per_comment = working.groupby("comment_id").size()
    kept_comments = counts_per_comment[counts_per_comment >= min_annotations_per_item].index
    working = working[working["comment_id"].isin(kept_comments)]
    items = items.loc[items.index.intersection(kept_comments)]

    if len(items) == 0:
        raise ValueError("No items remain after filtering.")

    # Stratified-by-S sub-sampling (preserves sensitive group balance).
    if max_items is not None and max_items > 0 and len(items) > max_items:
        n_target_s1 = int(round(max_items * items["S"].mean()))
        n_target_s0 = max_items - n_target_s1

        s1_pool = items.index[items["S"] == 1].to_numpy()
        s0_pool = items.index[items["S"] == 0].to_numpy()

        n_s1 = min(n_target_s1, len(s1_pool))
        n_s0 = min(n_target_s0, len(s0_pool))

        sampled_s1 = rng.choice(s1_pool, size=n_s1, replace=False) if n_s1 > 0 else np.array([], dtype=s1_pool.dtype)
        sampled_s0 = rng.choice(s0_pool, size=n_s0, replace=False) if n_s0 > 0 else np.array([], dtype=s0_pool.dtype)

        sampled_comments = pd.Index(np.concatenate([sampled_s1, sampled_s0]))
        items = items.loc[sampled_comments]
        working = working[working["comment_id"].isin(sampled_comments)]

    answers = working.pivot_table(
        index="comment_id",
        columns="annotator_id",
        values="annot_label",
        aggfunc="first",
    ).astype(float)

    answers = iterative_filter_answers(
        answers,
        min_annotations_per_item=min_annotations_per_item,
        min_annotations_per_annotator=min_annotations_per_annotator,
    )

    if answers.empty:
        raise ValueError("Empty answers matrix after filtering.")

    # Defensive: clip to {0, 1, NaN}
    answers = answers.where(answers.isna(), (answers > 0).astype(float))

    items = items.loc[answers.index]

    if items["S"].nunique() < 2:
        raise ValueError(f"Only one S group remains for {s_col}.")
    if items["Y"].nunique() < 2:
        raise ValueError("Only one Y class remains.")

    s = pd.DataFrame(items["S"].astype(int).values, index=answers.index, columns=["s"])
    y = pd.DataFrame(items["Y"].astype(int).values, index=answers.index, columns=["y"])
    x = pd.DataFrame({"text_len": items["text_len"].astype(float).values}, index=answers.index)

    df = FairCrowdDataset(answers=answers, s=s, x=x, y=y)

    ann_per_item = answers.notna().sum(axis=1)
    ann_per_annot = answers.notna().sum(axis=0)

    diagnostics: Dict[str, object] = {
        "s_col": s_col,
        "annotation_col": annotation_col,
        "annotation_threshold": annotation_threshold,
        "y_threshold": y_threshold,
        "n_items": int(len(answers)),
        "n_annotators": int(answers.shape[1]),
        "n_annotations": int(answers.notna().sum().sum()),
        "mean_annotations_per_item": float(ann_per_item.mean()),
        "median_annotations_per_item": float(ann_per_item.median()),
        "mean_annotations_per_annotator": float(ann_per_annot.mean()),
        "median_annotations_per_annotator": float(ann_per_annot.median()),
        "p_s1": float(s["s"].mean()),
        "n_s1": int(s["s"].sum()),
        "n_s0": int((1 - s["s"]).sum()),
        "p_y1": float(y["y"].mean()),
        "p_y1_given_s1": float(y.loc[s["s"] == 1, "y"].mean()),
        "p_y1_given_s0": float(y.loc[s["s"] == 0, "y"].mean()),
        "abs_dp_gold_final": abs(
            float(y.loc[s["s"] == 1, "y"].mean()) - float(y.loc[s["s"] == 0, "y"].mean())
        ),
    }

    if save_processed_dir is not None:
        save_processed_dir.mkdir(parents=True, exist_ok=True)
        answers.to_csv(save_processed_dir / "mhs_answers.csv")
        s.to_csv(save_processed_dir / "mhs_s.csv")
        x.to_csv(save_processed_dir / "mhs_x.csv")
        y.to_csv(save_processed_dir / "mhs_y.csv")
        pd.DataFrame([diagnostics]).to_csv(save_processed_dir / "mhs_diagnostics.csv", index=False)
        report.to_csv(save_processed_dir / "mhs_target_imbalance_report.csv", index=False)

    return df, diagnostics, report



def build_algorithms(df: FairCrowdDataset, include_fairtd: bool = False):
    n_classes = int(np.max(df["y"].values) + 1)
    n_annotators = len(df.answers.values[0])

    algorithms = []
    if include_fairtd:
        algorithms.append(FairTD_ref())

    maj = Majority_vote(n_classes, n_annotators)
    reg_maj = Optimal_reg(maj); reg_maj.name = "FC_Maj"
    post_td_maj = FairTD_Post(maj); post_td_maj.name = "Post_TD_Maj"

    geo_for_fc = Geometric(n_classes, n_annotators)
    reg_bayes = Optimal_reg(geo_for_fc); reg_bayes.name = "FC_Bayes_Count"

    geo_for_post = Geometric(n_classes, n_annotators)
    post_td_bayes = FairTD_Post(geo_for_post); post_td_bayes.name = "Post_TD_Bayes_Count"

    # DS-Multiple is unsupervised; fitting on full df does not leak Y.
    dwm = DawidSkeneMultiple()
    _ = dwm.run(df)
    reg_DS = Optimal_reg(dwm); reg_DS.name = "FC_DS"
    post_td_DS = FairTD_Post(dwm); post_td_DS.name = "Post_TD_DS"

    algorithms.extend([
        post_td_bayes, reg_bayes,
        reg_maj, post_td_maj,
        reg_DS, post_td_DS,
    ])
    return algorithms


# ============================================================
# Plotting / summaries
# ============================================================

def save_summary_tables(result: pd.DataFrame, output_dir: Path):
    metric_cols = [c for c in [
        "accuracy", "f1_score",
        "DemographicParity_", "EqualOpportunities_", "PredictiveParity_",
        "execution_time",
    ] if c in result.columns]

    summary = (
        result.groupby(["algorithm", "epsilon", "train_size"], as_index=False)
        .agg({c: ["mean", "std"] for c in metric_cols})
    )
    summary.columns = [
        "_".join([x for x in col if x]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(output_dir / "mhs_summary_by_algorithm_epsilon.csv", index=False)
    return summary

def plot_f1_dp_with_legend(result, output_dir, filename_prefix="mhs_f1_vs_dp_with_legend"):
    agg = (
        result.groupby(["algorithm", "epsilon"], as_index=False)
        .agg(
            dp_mean=("DemographicParity_", "mean"),
            f1_mean=("f1_score", "mean"),
            f1_std=("f1_score", "std"),
        )
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    algo_list = sorted(agg["algorithm"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    n_algos = len(algo_list)

    # Range pour jitter
    xrange = agg["dp_mean"].max() - agg["dp_mean"].min()
    jitter_amp = 0.004 * xrange  # 0.4% du range

    for i, algo in enumerate(algo_list):
        subdf = agg[agg["algorithm"] == algo].sort_values("epsilon")

        ls = "-" if algo.startswith("FC") else "--"
        label = f"{algo} (Ours)" if algo.startswith("FC") else algo

        # Petit jitter horizontal pour séparer les courbes
        jitter = (i - (n_algos - 1) / 2) * jitter_amp
        x = subdf["dp_mean"].values + jitter
        y = subdf["f1_mean"].values
        ystd = subdf["f1_std"].fillna(0).values

        line, = ax.plot(
            x, y,
            linestyle=ls, marker=markers[i % len(markers)],
            markersize=7, linewidth=2, label=label,
        )
        ax.fill_between(x, y - ystd, y + ystd, alpha=0.15,
                        color=line.get_color())

        # Annoter seulement eps min et max, sur l'algo le plus à gauche du nuage
        eps_min = subdf["epsilon"].min()
        eps_max = subdf["epsilon"].max()

        # Décalage vertical par algo pour eviter chevauchement labels
        yshift = (i - (n_algos - 1) / 2) * 7  # 7 px par algo

        for _, row in subdf.iterrows():
            if row["epsilon"] not in (eps_min, eps_max):
                continue
            label_eps = (rf"$\epsilon={row['epsilon']:.2f}$"
                         if row["epsilon"] == eps_min else f"{row['epsilon']:.2f}")
            ax.annotate(
                label_eps,
                (row["dp_mean"] + jitter, row["f1_mean"]),
                fontsize=7,
                xytext=(5, yshift),
                textcoords="offset points",
                color=line.get_color(),
                alpha=0.9,
            )

    ax.set_xlabel("Demographic Parity gap")
    ax.set_ylabel("F1 score")
    ax.set_title("Fairness–Accuracy Trade-off")
    ax.legend(
        title="Algorithm", fontsize=8, title_fontsize=9,
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True,
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    #ax.set_xlim(left=0)

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{filename_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
#def plot_f1_dp_with_legend(result: pd.DataFrame, output_dir: Path,
#                            filename_prefix: str = "mhs_f1_vs_dp_with_legend"):
#    agg = (
#        result.groupby(["algorithm", "epsilon"], as_index=False)
#        .agg(
#            dp_mean=("DemographicParity_", "mean"),
#            dp_std=("DemographicParity_", "std"),
#            f1_mean=("f1_score", "mean"),
#            f1_std=("f1_score", "std"),
#        )
#    )

#    fig, ax = plt.subplots(figsize=(7.5, 5.0))
#    for algo, subdf in agg.groupby("algorithm"):
#        subdf = subdf.sort_values("epsilon")
#        linestyle = "-" if algo.startswith("FC") else "--"
#        label = f"{algo} (Ours)" if algo.startswith("FC") else algo
#        x = subdf["dp_mean"].values
#        y = subdf["f1_mean"].values
#        y_std = subdf["f1_std"].fillna(0).values
#        ax.plot(x, y, linestyle=linestyle, marker="o", linewidth=2, label=label)
#        ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)
#        for _, row in subdf.iterrows():
#            ax.annotate(f"{row['epsilon']:.2f}", (row["dp_mean"], row["f1_mean"]),
#                        fontsize=8, xytext=(3, 3), textcoords="offset points")

#    ax.set_xlabel("Demographic Parity gap")
#    ax.set_ylabel("F1 score")
#    ax.set_title("Measuring Hate Speech — Fairness–Accuracy Trade-off")
#    ax.legend(title="Algorithm", fontsize=8, title_fontsize=9,
#              loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
#    ax.grid(True, linestyle="--", alpha=0.3)
    #ax.set_xlim(left=0)
#    fig.tight_layout(rect=[0, 0, 0.78, 1])
#    output_dir.mkdir(parents=True, exist_ok=True)
#    fig.savefig(output_dir / f"{filename_prefix}.png", dpi=300, bbox_inches="tight")
#    fig.savefig(output_dir / f"{filename_prefix}.pdf", bbox_inches="tight")
#    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--include-narrow-targets", action="store_true",
                        help="Include narrow target_* columns in the report (default: only broad).")

    parser.add_argument("--s-col", type=str, default="auto")
    parser.add_argument("--y-threshold", type=float, default=0.5)
    parser.add_argument("--annotation-col", type=str, default="hatespeech")
    parser.add_argument("--annotation-threshold", type=float, default=0.5)

    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--max-annotators", type=int, default=800)
    parser.add_argument("--min-annotations-per-item", type=int, default=5)
    parser.add_argument("--min-annotations-per-annotator", type=int, default=5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-repet", type=int, default=10)
    parser.add_argument("--train-size", type=float, default=0.4)
    parser.add_argument("--eps-list", type=float, nargs="+",
                        default=[0.05, 0.1, 0.15 ,0.2])

    parser.add_argument("--include-fairtd", action="store_true")
    parser.add_argument("--output-dir", type=str, default="figures_measuring_hate_speech")
    parser.add_argument("--save-processed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    only_broad = not args.include_narrow_targets

    print("Loading Measuring Hate Speech...")
    raw = load_raw_measuring_hate_speech()
    print("Raw shape:", raw.shape)

    report = build_target_imbalance_report(raw, y_threshold=args.y_threshold, only_broad=only_broad)
    report_path = output_dir / "mhs_target_imbalance_report.csv"
    report.to_csv(report_path, index=False)

    print("\nTarget columns ranked by DP gold gap (descending):")
    print(report[["s_col", "n_items", "n_s1", "p_s1", "p_y1",
                  "p_y1_given_s1", "p_y1_given_s0", "abs_dp_gold"]].head(25).to_string(index=False))

    if args.diagnose_only:
        suggested = choose_s_col_auto(report)
        print(f"\nSuggested s_col: {suggested}")
        return

    save_processed_dir = output_dir / "processed" if args.save_processed else None
    df, diagnostics, _ = build_mhs_faircrowd_dataset(
        raw=raw, s_col=args.s_col, y_threshold=args.y_threshold,
        annotation_col=args.annotation_col, annotation_threshold=args.annotation_threshold,
        max_items=args.max_items, max_annotators=args.max_annotators,
        min_annotations_per_item=args.min_annotations_per_item,
        min_annotations_per_annotator=args.min_annotations_per_annotator,
        random_state=args.seed, only_broad=only_broad,
        save_processed_dir=save_processed_dir,
    )

    pd.DataFrame([diagnostics]).to_csv(output_dir / "mhs_dataset_diagnostics.csv", index=False)
    print("\nFinal dataset diagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")

    algorithms = build_algorithms(df, include_fairtd=args.include_fairtd)

    accuracy_metrics = [accuracy, f1_score]
    fairness_metrics = [DemographicParity_(), EqualOpportunities_(), PredictiveParity_()]

    print("\nRunning benchmark...")
    result = evaluate_algorithms(
        df, algorithms, args.eps_list, accuracy_metrics, fairness_metrics,
        train_sizes_list=[args.train_size], n_repet=args.n_repet,
        filename_prefix="mhs_exp", save_dir="faircrowd/benchmark_new/exp_save",
        full_test=False,
    )

    result.to_csv(output_dir / "mhs_raw_results.csv", index=False)
    summary = save_summary_tables(result, output_dir)
    print("\nSummary head:")
    print(summary.head(30).to_string(index=False))

    plot_f1_vs_demographic_parity_with_variance(
        result, save_dir=str(output_dir), filename_prefix="mhs_repo_f1_vs_dp",
    )
    plot_f1_dp_with_legend(result, output_dir=output_dir,
                            filename_prefix="mhs_f1_vs_dp_with_legend")
    print("\nDone. Outputs in:", output_dir.resolve())


if __name__ == "__main__":
    main()