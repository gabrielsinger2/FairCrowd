"""
Run synthetic fair-crowdsourcing experiments with correlated annotators.

This script is intentionally separated from run_experiments_synhtetic.py so that the
original independent-annotator experiment remains unchanged.

Before running this file, make sure that generate_synthetic_data_correlated.py is
located at the root of the repo, next to this file.

Run:
    python run_experiments_correlated.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from faircrowd.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    false_positive_rate,
    false_negative_rate,
    DemographicParity_,
    EqualOpportunities_,
    PredictiveParity_,
)
from faircrowd.utils.exploratory import *
from faircrowd.competitors.fair_td import *
from faircrowd.competitors.geom_model import *
from faircrowd.benchmark_new.comparison import *
from faircrowd.competitors import (
    FairTD,
    FairTD_ref,
    Optimal_reg,
    Geometric,
    Majority_gold_op,
)
from faircrowd.truth_inference.dawid_skene_multiple import DawidSkeneMultiple

from data_generators.generate_synthetic_data_correlated import (
    generate_synthetic_binary_correlated_restricted,
    empirical_error_correlation,
)



N = 2000
R = 100

R_annots_max = 5

bias_toward_1 = 0.2
prop_s_1 = 0.6

# Error-rate ranges by sensitive group.
# rho_0 controls annotator error rates on S=0.
# rho_1 controls annotator error rates on S=1.
rho_0 = (0.0, 0.5)
rho_1 = (0.0, 0.4)

# Correlation strength between annotators' errors.
# 0.0 means conditionally independent annotators.
# 1.0 means strong item-level shared errors.
corr_0 = 0.5
corr_1 = 0.5

seed = 0

figures_dir = "figures_synthetic_correlated"
filename_prefix = "1_correlated_synhtetic_exp"

Path(figures_dir).mkdir(parents=True, exist_ok=True)


# ============================================================
# Generate correlated synthetic dataset
# ============================================================

Y_annotators, S, Y, confusion, prop, details = generate_synthetic_binary_correlated_restricted(
    N=N,
    R=R,
    R_anot_max=R_annots_max,
    bias_toward_1=bias_toward_1,
    prop_s_1=prop_s_1,
    rho_0=rho_0,
    rho_1=rho_1,
    corr_0=corr_0,
    corr_1=corr_1,
    seed=seed,
    return_details=True,
)

df = FairCrowdDataset(
    pd.DataFrame(Y_annotators),
    pd.DataFrame(S[:, None]),
    pd.DataFrame(np.zeros(len(Y_annotators))),
    pd.DataFrame(Y[:, None]),
)

print("============================================================")
print("Correlated synthetic dataset")
print("============================================================")
print("dataset size:", len(df.df))
print("number annotators:", len(df.answers.values[0]))
print("max annotations per item:", R_annots_max)
print("prop:", prop)
print("corr_0:", corr_0)
print("corr_1:", corr_1)

# Empirical pairwise correlation of annotator error indicators.
# This uses the synthetic ground truth Y and ignores np.nan values.
corr_errors = empirical_error_correlation(Y_annotators, Y)
off_diag = corr_errors[~np.eye(corr_errors.shape[0], dtype=bool)]
off_diag = off_diag[~np.isnan(off_diag)]

print("mean empirical error correlation:", float(np.mean(off_diag)))
print("median empirical error correlation:", float(np.median(off_diag)))
print("min empirical error correlation:", float(np.min(off_diag)))
print("max empirical error correlation:", float(np.max(off_diag)))
print("============================================================")


# ============================================================
# Prepare arrays
# ============================================================

n_classes_ = int(np.max(df["y"].values) + 1)
n_annotators = len(df.answers.values[0])

train_i, test_i = train_test_split(
    df._df.index,
    test_size=0.4,
    shuffle=True,
    random_state=0,
)

df_test_answers = df.answers.loc[test_i].values
df_test_s = df.s.loc[test_i].values.flatten()
df_test_y = df.y.loc[test_i].values.flatten()

df_answers = df.answers.values
df_s = df.s.values.flatten()
df_y = df.y.values.flatten()


# ============================================================
# Truth inference / competitors
# ============================================================

# Dawid-Skene multiple
dwm = DawidSkeneMultiple()
out = dwm.run(df)

reg_DS = Optimal_reg(dwm)
reg_DS.name = "FC_DS"

post_td_DS = FairTD_Post(dwm)
post_td_DS.name = "Post_TD_DS"


# Geometric/Bayes model
geo = Geometric(n_classes_, n_annotators)
geo.fit(df_answers, df_s, df_y)

reg_bayes = Optimal_reg(geo)
reg_bayes.name = "FC_Bayes"

post_td_bayes = FairTD_Post(geo)
post_td_bayes.name = "Post_TD_Bayes"


# Majority vote
maj = Majority_vote(n_classes_, n_annotators)

reg_maj = Optimal_reg(maj)
reg_maj.name = "FC_Maj"

post_td_maj = FairTD_Post(maj)
post_td_maj.name = "Post_TD_Maj"


# ============================================================
# Benchmark
# ============================================================

td_algorithms = [
    FairTD_ref(),
    post_td_bayes,
    reg_bayes,
    reg_maj,
    post_td_maj,
    reg_DS,
    post_td_DS,
]

eps_list = [0.01, 0.05, 0.1, 0.2]

accuracy_metrics = [accuracy, f1_score]
fairness_metrics = [
    DemographicParity_(),
    EqualOpportunities_(),
    PredictiveParity_(),
]

train_sizes_list = [0.4]

d_data = evaluate_algorithms(
    df,
    td_algorithms,
    eps_list,
    accuracy_metrics,
    fairness_metrics,
    train_sizes_list=train_sizes_list,
    n_repet=10,
    filename_prefix=filename_prefix,
    full_test=False,
)


# ============================================================
# Plot
# ============================================================

plot_f1_vs_demographic_parity_with_variance(
    d_data,
    save_dir=figures_dir,
)

print("Done.")
print("Figures saved in:", figures_dir)
