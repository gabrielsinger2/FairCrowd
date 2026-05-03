# FairCrowd: Fairness-Aware Crowdsourced Label Aggregation 

##  🎉 Accepted at ICML 2026 🎓✨
> Authors: Gabriel Singer, Samuel Gruffaz, Vo Van Olivier, Vayatis Nicolas, Kalogeratos Argyris — ICML 2026.

FairCrowd is a **post-processing algorithm** that takes any crowdsourced label aggregator: Majority Vote, Bayesian, Dawid–Skene, and turns it into a strictly **ε-fair** aggregator under a Demographic Parity constraint, with no loss of generality on the upstream method.

In one line: *if you already have a way to aggregate noisy crowd labels, FairCrowd makes it fair, with provable guarantees and a 2-second runtime.*

## Installation

```bash
git clone https://github.com/gabrielsinger2/FairCrowd.git
cd FairCrowd
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install datasets pyarrow   # for Hugging Face datasets
```

Requires Python 3.10+. All experiments run on CPU and finish in a few minutes (FairCrowd itself takes less than 2 seconds per ε on real datasets).

---

## Quickstart

```python
from faircrowd.competitors import Optimal_reg, Geometric  # FairCrowd + Bayes count
from faircrowd.metrics import DemographicParity_, f1_score

# any aggregator that produces posterior probabilities Φ̂(w, a)
bayes = Geometric(n_classes=2, n_annotators=R)
bayes.fit(answers, sensitive, gold)

# wrap it with FairCrowd to enforce ε = 0.05 demographic parity
fair_aggregator = Optimal_reg(bayes)
out = fair_aggregator.run(df, eps=0.05)

print(f"F1:      {f1_score(df['y'], out.labels):.3f}")
print(f"DP gap:  {DemographicParity_().compute(out.labels, df['s'], df['y']):.3f}")
```

---

## Reproducing the paper experiments:

### Comparison experiments (Figures 3 and 4)

Fairness/accuracy trade-off at varying `ε`, against the in-processing baseline FairTD and the post-processing baseline Post_TD from Li et al. (2020).

```bash
python scripts/run_experiments_synthetic.py
python scripts/run_experiments_crowd_judgement.py
python scripts/run_experiments_jigsaw.py
```

Each script:
1. loads the dataset via `faircrowd.datasets`,
2. fits the upstream aggregators (Majority Vote, Bayesian, Dawid–Skene),
3. applies FairCrowd and Post_TD at ε ∈ {0.01, 0.05, 0.10, 0.20},
4. computes F1, accuracy, and DP/EOpp/PP across 10 seeds with 60% test resampling,
5. saves raw results to `faircrowd/benchmark_new/exp_save/` and figures to `figures_*/`.

### Quick `make` shortcuts

```bash
make synthetic
make synthetic-correlated
make compare
make mhs-diagnose
```

---

## Repository layout

```
FairCrowd/
├── faircrowd/                  # core library
│   ├── core/                   #   FairCrowdDataset, base classes, metrics
│   ├── competitors/            #   Optimal_reg (FairCrowd), Geometric (Bayes), FairTD_Post
│   ├── truth_inference/        #   Dawid-Skene, Majority Voting, GLAD
│   ├── metrics/                #   DemographicParity_, EqualOpportunities_, PredictiveParity_
│   ├── benchmark_new/          #   evaluate_algorithms, plot helpers
│   └── datasets/               #   Crowd Judgement, Jigsaw Toxicity, Synthetic loaders
├── data_generators/            # synthetic data generation (independent + correlated noise)
├── scripts/                    # reproducible experiment scripts
├── configs/                    # YAML configs for each experiment
├── outputs/                    # raw results, summary tables, figures
├── exploratory/                # notebooks and exploratory scripts
└── legacy/                     # archived / superseded scripts
```

> **Naming note:** `Optimal_reg` is **FairCrowd** (Algorithm 1 in the paper); `Geometric` is the **Bayesian aggregator** (`ϕ*` in the paper) when fitted by counting. Renaming for clarity will happen in a future release; current names are preserved for git history compatibility.

---

## Algorithms implemented

| Class                    | Paper name             | Role                                                       |
|--------------------------|------------------------|------------------------------------------------------------|
| `Majority_vote`          | `ϕ_MV`                 | Simple majority aggregation baseline                       |
| `Geometric`              | `ϕ*` (Bayes)           | Bayes-optimal aggregator with one-coin model               |
| `DawidSkene`             | `ϕ_DS`                 | EM-based aggregator (no ground truth needed)               |
| `DawidSkeneMultiple`     | `ϕ_DS` (per group)     | Group-conditional Dawid–Skene                              |
| **`Optimal_reg`**        | **FairCrowd**          | **ε-fair post-processing of any of the above (Algorithm 1)** |
| `FairTD_Post`            | Post_TD                | Post-processing baseline from Li et al. (2020)             |
| `FairTD_ref`             | FairTD                 | In-processing baseline from Li et al. (2020)               |

---

## Datasets

Three benchmarks are proposed in `faircrowd/datasets/`:

- **Synthetic**: 2000 tasks × 100 annotators, configurable annotator skill distributions and sensitive feature imbalance.
- **Crowd Judgement** (Dressel & Farid, 2018): 1000 COMPAS cases × 20 annotators each. Sensitive = race. Task: recidivism prediction.
- **Jigsaw Toxicity**: 5000 civil comments × heavy-tailed annotator distribution (median 4, std 75). Sensitive = mention of a discriminated identity group. Task: toxicity classification.

Optional benchmarks (Hugging Face):
- **Measuring Hate Speech** (Berkeley DLab) via `scripts/run_experiments_measuring_hate_speech.py`.
- **Social Bias Frames** (AllenAI) via `scripts/run_experiments_sbic.py`.

---

## Citation

add soon

---

## Authors and contact

- **Gabriel Singer** — SNCF and ENS Paris-Saclay: gabrielsinger2@gmail.com
- **Samuel Gruffaz** — [ENS Paris Saclay]

For questions about the code, please open a GitHub issue. For research collaborations: *gabrielsinger2@gmail.com*.
