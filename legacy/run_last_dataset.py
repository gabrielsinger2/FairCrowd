"""
Run FC post-processing experiments on the Social Bias Frames dataset.

Dataset:
    allenai/social_bias_frames    (SBIC v2, ACL 2020, CC-BY-4.0)
    https://huggingface.co/datasets/allenai/social_bias_frames

Why this dataset is well-suited:
    - Real crowdsourcing: each post has multiple annotators (worker_id available
      via the 'WorkerId' column in the underlying CSVs, exposed by HF).
    - Multiple natural binary fairness axes:
        * targetMinority: whether the post targets a minority group
          (we use the boolean version derived from the column).
        * targetCategory: gender / race / culture / ... (we collapse to binary).
    - Label of interest: offensiveYN (0 / 0.5 / 1) -> binarize to {0, 1}.

Setup:
    pip install datasets pyarrow
    (datasets >= 4.0 is fine; we use the auto-converted parquet revision.)

Usage:
    python run_experiments_sbic.py --diagnose-only
    python run_experiments_sbic.py --max-items 5000 --max-annotators 600 --n-repet 10

Loader strategy (in order):
    1. HF `datasets` with revision="refs/convert/parquet" — works on
       datasets >= 4.0 because it bypasses the (now-unsupported) script.
    2. HF `datasets` default config — only works on datasets < 4.0.
    3. Direct download of the official SBIC v2 CSVs from
       https://maartensap.com/social-bias-frames/SBIC.v2.tgz
       (independent of the HF stack; works with any datasets version).

Notes:
    HF's SBF release may not expose WorkerId on all configs. We fall back to
    a per-(post, annotation row index modulo K) annotator if WorkerId is not
    available — this is just a sanity fallback and a warning is printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from faircrowd.core import FairCrowdDataset
from faircrowd.metrics import (
    accuracy, f1_score,
    DemographicParity_, EqualOpportunities_, PredictiveParity_,
)
from faircrowd.competitors.fair_td import FairTD_Post
from faircrowd.competitors.geom_model import Majority_vote
from faircrowd.benchmark_new.comparison import (
    evaluate_algorithms, plot_f1_vs_demographic_parity_with_variance,
)
from faircrowd.competitors import FairTD_ref, Optimal_reg, Geometric
from faircrowd.truth_inference.dawid_skene_multiple import DawidSkeneMultiple


# ============================================================
# Load
# ============================================================

def _load_raw_sbic_via_hf() -> Optional[pd.DataFrame]:
    """
    Try HF `datasets` with two strategies:
      A. The canonical repo (works only with datasets<4.0 because the repo
         still ships a script `social_bias_frames.py`).
      B. The auto-generated parquet revision `refs/convert/parquet`,
         which works with datasets>=4.0 since it bypasses the script.

    Returns the concatenated DataFrame, or None if both attempts fail.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[INFO] `datasets` not installed; skipping HF loader.")
        return None

    attempts = [
        # (name, kwargs, label)
        ("allenai/social_bias_frames",
         {"revision": "refs/convert/parquet"},
         "HF parquet revision"),
        ("allenai/social_bias_frames", {}, "HF default (script)"),
        ("social_bias_frames", {}, "HF legacy alias (script)"),
    ]

    for name, kwargs, label in attempts:
        try:
            print(f"[INFO] Trying {label}: {name} {kwargs}")
            ds = load_dataset(name, **kwargs)
            parts = []
            for split in ("train", "validation", "test"):
                if split in ds:
                    parts.append(ds[split].to_pandas())
            if parts:
                df = pd.concat(parts, axis=0, ignore_index=True)
                print(f"[INFO] Loaded via {label}; shape={df.shape}")
                return df
        except Exception as e:
            print(f"[WARN] {label} failed: {e!r}")

    return None


def _load_raw_sbic_via_direct_download(cache_dir: Path) -> pd.DataFrame:
    """
    Final fallback: download the original SBIC v2 CSVs directly from
    Maarten Sap's homepage. This avoids the HF datasets stack entirely.

    Mirror used:
        https://maartensap.com/social-bias-frames/SBIC.v2.tgz

    The archive contains:
        SBIC.v2.trn.csv
        SBIC.v2.dev.csv
        SBIC.v2.tst.csv
    with the documented schema (WorkerId, HITId, post, targetMinority,
    targetCategory, offensiveYN, intentYN, sexYN, ...).
    """
    import io, tarfile, urllib.request

    url = "https://maartensap.com/social-bias-frames/SBIC.v2.tgz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "SBIC.v2.tgz"

    if not archive_path.exists():
        print(f"[INFO] Downloading SBIC v2 from {url} ...")
        try:
            urllib.request.urlretrieve(url, archive_path)
        except Exception as e:
            raise RuntimeError(
                f"Direct download failed ({e!r}).\n"
                "Workaround: install an older datasets version, e.g.\n"
                '    pip install "datasets<4.0"\n'
                "and rerun this script."
            ) from e
    else:
        print(f"[INFO] Reusing cached archive at {archive_path}")

    parts = []
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if not member.name.endswith(".csv"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            buf = io.BytesIO(f.read())
            try:
                part = pd.read_csv(buf, low_memory=False)
            except Exception as e:
                print(f"[WARN] Could not parse {member.name}: {e!r}")
                continue
            parts.append(part)

    if not parts:
        raise RuntimeError("SBIC v2 archive contained no readable CSVs.")

    df = pd.concat(parts, axis=0, ignore_index=True)
    print(f"[INFO] Loaded via direct CSV download; shape={df.shape}")
    return df


def load_raw_sbic(cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Social Bias Frames as a single pandas DataFrame.

    Strategy:
      1. Try HF `datasets` (parquet revision, then script-based).
      2. Fall back to direct CSV download from the official mirror.

    The returned DataFrame keeps the original SBF schema, including
    WorkerId, HITId, post, targetMinority, targetCategory, offensiveYN.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "sbic_v2"

    df = _load_raw_sbic_via_hf()
    if df is not None and len(df) > 0:
        return df

    print("[INFO] Falling back to direct download.")
    return _load_raw_sbic_via_direct_download(cache_dir)


# ============================================================
# Annotator id resolution
# ============================================================

def resolve_annotator_col(raw: pd.DataFrame) -> str:
    candidates = ["WorkerId", "worker_id", "annotator_id", "annotatorid"]
    for c in candidates:
        if c in raw.columns:
            return c
    print("[WARN] No worker id column found; falling back to row-cycle annotators.")
    raw = raw.copy()
    raw["_pseudo_annot"] = (np.arange(len(raw)) % 200).astype(int)
    return "_pseudo_annot"


def resolve_item_col(raw: pd.DataFrame) -> str:
    for c in ("HITId", "post_id", "id", "comment_id"):
        if c in raw.columns:
            return c
    if "post" in raw.columns:
        return "post"
    raise ValueError("No item id / post column found in SBF.")


# ============================================================
# Sensitive attribute and label
# ============================================================

def to_binary_offensive(values) -> pd.Series:
    """offensiveYN: typically 0.0 / 0.5 / 1.0 in SBF. Binarize at >0.5 strict."""
    v = pd.to_numeric(values, errors="coerce")
    return (v > 0.5).astype(float)


def to_binary_target_minority(values) -> pd.Series:
    """targetMinority: nonempty string => post targets a minority => S=1."""
    s = values.fillna("").astype(str).str.strip()
    return (s.str.len() > 0).astype(int)


def build_sbic_faircrowd_dataset(
    raw: pd.DataFrame,
    s_col_choice: str = "targetMinority",
    label_col: str = "offensiveYN",
    max_items: int = 5000,
    max_annotators: int = 600,
    min_annotations_per_item: int = 3,
    min_annotations_per_annotator: int = 5,
    random_state: int = 0,
    save_processed_dir: Optional[Path] = None,
) -> Tuple[FairCrowdDataset, Dict[str, object]]:
    rng = np.random.default_rng(random_state)

    annot_col = resolve_annotator_col(raw)
    item_col = resolve_item_col(raw)

    if label_col not in raw.columns:
        raise ValueError(f"label_col {label_col!r} missing.")
    if s_col_choice not in raw.columns:
        raise ValueError(f"s_col {s_col_choice!r} missing.")

    working = raw.copy()
    working["annot_label"] = to_binary_offensive(working[label_col])
    working = working.dropna(subset=[item_col, annot_col, "annot_label"])

    # Item-level table: aggregate S and Y by majority-of-annotators (since
    # in SBF, S and Y are per-annotator judgments).
    item_S = (
        working.groupby(item_col)[s_col_choice]
        .apply(lambda v: int(to_binary_target_minority(v).mean() > 0.5))
        .rename("S")
    )
    item_Y = (
        working.groupby(item_col)["annot_label"]
        .apply(lambda v: int(np.nanmean(v) > 0.5))
        .rename("Y")
    )
    items = pd.concat([item_S, item_Y], axis=1).dropna()

    # Pre-filter top annotators.
    annot_counts = working.groupby(annot_col).size().sort_values(ascending=False)
    if max_annotators is not None and max_annotators > 0:
        kept_annot = annot_counts.head(max_annotators).index
        working = working[working[annot_col].isin(kept_annot)]

    counts_per_item = working.groupby(item_col).size()
    kept_items = counts_per_item[counts_per_item >= min_annotations_per_item].index
    working = working[working[item_col].isin(kept_items)]
    items = items.loc[items.index.intersection(kept_items)]

    if len(items) == 0:
        raise ValueError("Empty after filtering.")

    # Stratified subsample by S.
    if max_items and len(items) > max_items:
        n_target_s1 = int(round(max_items * items["S"].mean()))
        n_target_s0 = max_items - n_target_s1
        s1_pool = items.index[items["S"] == 1].to_numpy()
        s0_pool = items.index[items["S"] == 0].to_numpy()
        n_s1 = min(n_target_s1, len(s1_pool))
        n_s0 = min(n_target_s0, len(s0_pool))
        chosen_s1 = rng.choice(s1_pool, size=n_s1, replace=False) if n_s1 > 0 else np.array([], dtype=s1_pool.dtype)
        chosen_s0 = rng.choice(s0_pool, size=n_s0, replace=False) if n_s0 > 0 else np.array([], dtype=s0_pool.dtype)
        chosen = pd.Index(np.concatenate([chosen_s1, chosen_s0]))
        items = items.loc[chosen]
        working = working[working[item_col].isin(chosen)]

    answers = working.pivot_table(
        index=item_col, columns=annot_col, values="annot_label", aggfunc="first"
    ).astype(float)

    # Iterative filter
    for _ in range(10):
        old = answers.shape
        answers = answers.loc[answers.notna().sum(axis=1) >= min_annotations_per_item]
        answers = answers.loc[:, answers.notna().sum(axis=0) >= min_annotations_per_annotator]
        if answers.shape == old:
            break

    if answers.empty:
        raise ValueError("Empty answers after filtering.")

    answers = answers.where(answers.isna(), (answers > 0).astype(float))
    items = items.loc[answers.index]

    if items["S"].nunique() < 2 or items["Y"].nunique() < 2:
        raise ValueError("Degenerate S or Y after filtering.")

    s = pd.DataFrame(items["S"].astype(int).values, index=answers.index, columns=["s"])
    y = pd.DataFrame(items["Y"].astype(int).values, index=answers.index, columns=["y"])
    x = pd.DataFrame({"dummy": np.zeros(len(answers))}, index=answers.index)

    df = FairCrowdDataset(answers=answers, s=s, x=x, y=y)

    diagnostics = {
        "s_col": s_col_choice,
        "label_col": label_col,
        "n_items": int(len(answers)),
        "n_annotators": int(answers.shape[1]),
        "n_annotations": int(answers.notna().sum().sum()),
        "p_s1": float(s["s"].mean()),
        "p_y1": float(y["y"].mean()),
        "p_y1_given_s1": float(y.loc[s["s"] == 1, "y"].mean()),
        "p_y1_given_s0": float(y.loc[s["s"] == 0, "y"].mean()),
        "abs_dp_gold": abs(
            float(y.loc[s["s"] == 1, "y"].mean()) - float(y.loc[s["s"] == 0, "y"].mean())
        ),
        "mean_annot_per_item": float(answers.notna().sum(axis=1).mean()),
        "mean_annot_per_worker": float(answers.notna().sum(axis=0).mean()),
    }

    if save_processed_dir is not None:
        save_processed_dir.mkdir(parents=True, exist_ok=True)
        answers.to_csv(save_processed_dir / "sbic_answers.csv")
        s.to_csv(save_processed_dir / "sbic_s.csv")
        y.to_csv(save_processed_dir / "sbic_y.csv")
        pd.DataFrame([diagnostics]).to_csv(save_processed_dir / "sbic_diagnostics.csv", index=False)

    return df, diagnostics


# ============================================================
# Algorithms (same as MHS)
# ============================================================

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

    dwm = DawidSkeneMultiple(); _ = dwm.run(df)
    reg_DS = Optimal_reg(dwm); reg_DS.name = "FC_DS"
    post_td_DS = FairTD_Post(dwm); post_td_DS.name = "Post_TD_DS"

    algorithms.extend([
        post_td_bayes, reg_bayes,
        reg_maj, post_td_maj,
        reg_DS, post_td_DS,
    ])
    return algorithms


# ============================================================
# Plot helper (with legend)
# ============================================================

def plot_f1_dp_with_legend(result, output_dir: Path, filename_prefix: str = "sbic_f1_vs_dp_with_legend"):
    agg = (
        result.groupby(["algorithm", "epsilon"], as_index=False)
        .agg(
            dp_mean=("DemographicParity_", "mean"),
            f1_mean=("f1_score", "mean"),
            f1_std=("f1_score", "std"),
        )
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for algo, subdf in agg.groupby("algorithm"):
        subdf = subdf.sort_values("epsilon")
        ls = "-" if algo.startswith("FC") else "--"
        label = f"{algo} (Ours)" if algo.startswith("FC") else algo
        x = subdf["dp_mean"].values
        y = subdf["f1_mean"].values
        ystd = subdf["f1_std"].fillna(0).values
        ax.plot(x, y, linestyle=ls, marker="o", linewidth=2, label=label)
        ax.fill_between(x, y - ystd, y + ystd, alpha=0.2)
        for _, r in subdf.iterrows():
            ax.annotate(f"{r['epsilon']:.2f}", (r["dp_mean"], r["f1_mean"]),
                        fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Demographic Parity gap")
    ax.set_ylabel("F1 score")
    ax.set_title("Social Bias Frames — Fairness–Accuracy Trade-off")
    ax.legend(title="Algorithm", fontsize=8, title_fontsize=9,
              loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{filename_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--s-col", type=str, default="targetMinority")
    parser.add_argument("--label-col", type=str, default="offensiveYN")
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--max-annotators", type=int, default=600)
    parser.add_argument("--min-annotations-per-item", type=int, default=3)
    parser.add_argument("--min-annotations-per-annotator", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-repet", type=int, default=10)
    parser.add_argument("--train-size", type=float, default=0.4)
    parser.add_argument("--eps-list", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2])
    parser.add_argument("--include-fairtd", action="store_true")
    parser.add_argument("--output-dir", type=str, default="figures_sbic")
    parser.add_argument("--save-processed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print("Loading SBF...")
    raw = load_raw_sbic()
    print("Raw shape:", raw.shape, "columns:", list(raw.columns)[:20], "...")

    if args.diagnose_only:
        n = len(raw)
        s = to_binary_target_minority(raw[args.s_col]) if args.s_col in raw.columns else None
        y = to_binary_offensive(raw[args.label_col]) if args.label_col in raw.columns else None
        print(f"n_rows={n}")
        if s is not None:
            print(f"p_s1 (annot-level) = {s.mean():.3f}")
        if y is not None:
            print(f"p_y1 (annot-level) = {y.mean():.3f}")
        return

    save_processed_dir = out / "processed" if args.save_processed else None
    df, diag = build_sbic_faircrowd_dataset(
        raw=raw, s_col_choice=args.s_col, label_col=args.label_col,
        max_items=args.max_items, max_annotators=args.max_annotators,
        min_annotations_per_item=args.min_annotations_per_item,
        min_annotations_per_annotator=args.min_annotations_per_annotator,
        random_state=args.seed, save_processed_dir=save_processed_dir,
    )
    pd.DataFrame([diag]).to_csv(out / "sbic_dataset_diagnostics.csv", index=False)
    print("\nDiagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    algos = build_algorithms(df, include_fairtd=args.include_fairtd)
    accuracy_metrics = [accuracy, f1_score]
    fairness_metrics = [DemographicParity_(), EqualOpportunities_(), PredictiveParity_()]

    result = evaluate_algorithms(
        df, algos, args.eps_list, accuracy_metrics, fairness_metrics,
        train_sizes_list=[args.train_size], n_repet=args.n_repet,
        filename_prefix="sbic_exp", save_dir="faircrowd/benchmark_new/exp_save",
        full_test=False,
    )
    result.to_csv(out / "sbic_raw_results.csv", index=False)

    plot_f1_vs_demographic_parity_with_variance(
        result, save_dir=str(out), filename_prefix="sbic_repo_f1_vs_dp",
    )
    plot_f1_dp_with_legend(result, output_dir=out, filename_prefix="sbic_f1_vs_dp_with_legend")
    print("\nDone. Outputs in:", out.resolve())


if __name__ == "__main__":
    main()