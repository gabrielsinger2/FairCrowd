#!/usr/bin/env bash
# ============================================================
# Reorganize FairCrowd-ICML into a clean, reproducible layout.
# Run from the repo root only.
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# 0. Safety check: are we at the repo root?
# ------------------------------------------------------------

REPO_NAME_HINT="Fair_Crowd_ICML"  # adjust if your folder is named differently
EXPECTED_MARKERS=("faircrowd" "requirements.txt")

missing=0
for marker in "${EXPECTED_MARKERS[@]}"; do
    if [ ! -e "$marker" ]; then
        echo "[ABORT] Marker '$marker' not found in $(pwd)."
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "Run this script from the root of the FairCrowd repo."
    echo "Current dir: $(pwd)"
    exit 1
fi
echo "[OK] Running from $(pwd)"

# ------------------------------------------------------------
# 1. Cleanup junk files (safe operations only)
# ------------------------------------------------------------

find . -name "__pycache__" -type d -not -path "./.git/*" -prune -exec rm -rf {} + 2>/dev/null || true
find . -name ".DS_Store" -not -path "./.git/*" -delete 2>/dev/null || true
find . -name "*~" -not -path "./.git/*" -delete 2>/dev/null || true
find . -name "*.pyc" -not -path "./.git/*" -delete 2>/dev/null || true

# ------------------------------------------------------------
# 2. Create the target directory structure
# ------------------------------------------------------------

mkdir -p data_generators scripts configs
mkdir -p outputs/raw_results outputs/tables outputs/figures
mkdir -p paper_figures exploratory legacy
touch data_generators/__init__.py

# ------------------------------------------------------------
# 3. README — only create if absent (do NOT overwrite)
# ------------------------------------------------------------

if [ -f readme.txt ] && [ ! -f README.md ]; then
    mv readme.txt README.md
fi

if [ ! -f README.md ]; then
cat > README.md <<'EOF'
# FairCrowd-ICML

Research code for fairness-aware crowdsourced label aggregation under noisy annotations.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install datasets pyarrow
```

## Main experiments

```bash
make synthetic
make synthetic-correlated
make compare
make mhs-diagnose
```
EOF
fi

# ------------------------------------------------------------
# 4. LICENSE — only create stub if absent
# ------------------------------------------------------------

if [ ! -f LICENSE ]; then
cat > LICENSE <<'EOF'
TODO: choose a license (MIT, Apache-2.0, GPL-3.0, ...).
EOF
fi

# ------------------------------------------------------------
# 5. Move data generators
# ------------------------------------------------------------

move_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        mv "$src" "$dst"
        echo "  moved: $src -> $dst"
    elif [ -f "$src" ] && [ -f "$dst" ]; then
        echo "  [skip] both exist: $src and $dst (manual merge needed)"
    fi
}

echo "[5] Moving data generators..."
move_if_exists generate_synthetic_data.py            data_generators/generate_synthetic_data.py
move_if_exists generate_synthetic_data_correlated.py data_generators/generate_synthetic_data_correlated.py

# ------------------------------------------------------------
# 6. Move main scripts (with consistent naming)
# ------------------------------------------------------------

echo "[6] Moving main scripts..."
move_if_exists run_experiments_synhtetic.py             scripts/run_experiments_synthetic.py
move_if_exists run_experiments_synhtetic_convergence.py scripts/run_experiments_synthetic_convergence.py
move_if_exists run_experiments_synthetics_correlated.py scripts/run_experiments_synthetic_correlated.py
move_if_exists run_experiments_synthetic_correlated.py  scripts/run_experiments_synthetic_correlated.py
move_if_exists run_experiments_correlated.py            scripts/run_experiments_synthetic_correlated.py
move_if_exists run_compare_independent_vs_correlated.py scripts/run_compare_independent_vs_correlated.py
move_if_exists run_experiments_measuring_hate_speech.py scripts/run_experiments_measuring_hate_speech.py
move_if_exists run_experiments_sbic.py                  scripts/run_experiments_sbic.py
move_if_exists run_experiments_crowd_jugement.py        scripts/run_experiments_crowd_judgement.py
move_if_exists run_experiments_crowd_judgement.py       scripts/run_experiments_crowd_judgement.py

# ------------------------------------------------------------
# 7. Exploratory & legacy
# ------------------------------------------------------------

echo "[7] Moving exploratory & legacy..."
move_if_exists exploration_jigsaw.py exploratory/exploration_jigsaw.py
move_if_exists exploration_synth.py  exploratory/exploration_synth.py
move_if_exists run_visu.py           exploratory/run_visu.py
move_if_exists run_last_dataset.py        legacy/run_last_dataset.py
move_if_exists run_experiments_jigsaw.py  legacy/run_experiments_jigsaw.py
move_if_exists "run_experiments_jigsaw.py~" legacy/run_experiments_jigsaw.py.bak
move_if_exists "run_experiments_synhtetic.py~" legacy/run_experiments_synthetic.py.bak

# ------------------------------------------------------------
# 8. Reproducible configs (only if absent)
# ------------------------------------------------------------

write_config_if_absent() {
    local path="$1"
    local content="$2"
    if [ ! -f "$path" ]; then
        printf "%s" "$content" > "$path"
        echo "  wrote: $path"
    else
        echo "  [skip] $path already exists"
    fi
}

write_config_if_absent configs/synthetic_default.yaml '
experiment: synthetic
N: 2000
R: 100
R_annots_max: 5
bias_toward_1: 0.2
prop_s_1: 0.6
rho_0: [0.0, 0.5]
rho_1: [0.0, 0.4]
seed: 0
n_repet: 10
eps_list: [0.01, 0.05, 0.1, 0.2]
'

write_config_if_absent configs/correlated_default.yaml '
experiment: synthetic_correlated
N: 2000
R: 100
R_annots_max: 5
bias_toward_1: 0.2
prop_s_1: 0.6
rho_0: [0.0, 0.5]
rho_1: [0.0, 0.4]
corr_0: 0.5
corr_1: 0.5
seed: 0
n_repet: 10
eps_list: [0.01, 0.05, 0.1, 0.2]
'

write_config_if_absent configs/measuring_hate_speech_target_race.yaml '
experiment: measuring_hate_speech
s_col: target_race
annotation_threshold: 0.5
y_threshold: 0.5
max_items: 6000
max_annotators: 800
min_annotations_per_item: 5
min_annotations_per_annotator: 5
seed: 0
n_repet: 10
eps_list: [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]
'

# ------------------------------------------------------------
# 9. Patch imports in moved scripts
# ------------------------------------------------------------
# NOTE: heredoc is quoted ('PY') to avoid bash variable expansion of $1, etc.
#       Critical fix vs previous version: __file__ (double underscore) was
#       mangled by markdown rendering into **file**.
# ------------------------------------------------------------

echo "[9] Patching imports in scripts/..."
python3 - <<'PY'
from pathlib import Path

scripts_dir = Path("scripts")
if not scripts_dir.exists():
    raise SystemExit("No scripts/ directory found.")

root_block = (
    "from pathlib import Path\n"
    "import sys\n"
    "\n"
    "ROOT = Path(__file__).resolve().parents[1]\n"
    "if str(ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(ROOT))\n"
    "\n"
)

replacements = [
    ("from generate_synthetic_data import",
     "from data_generators.generate_synthetic_data import"),
    ("from generate_synthetic_data_correlated import",
     "from data_generators.generate_synthetic_data_correlated import"),
    ("import generate_synthetic_data_correlated",
     "import data_generators.generate_synthetic_data_correlated"),
    ("import generate_synthetic_data",
     "import data_generators.generate_synthetic_data"),
]

n_patched = 0
for path in scripts_dir.glob("*.py"):
    text = path.read_text()
    original = text

    if "ROOT = Path(__file__).resolve().parents[1]" not in text:
        text = root_block + text

    for old, new in replacements:
        text = text.replace(old, new)

    if text != original:
        path.write_text(text)
        n_patched += 1
        print(f"  patched: {path}")

print(f"Done. {n_patched} script(s) patched.")
PY

# ------------------------------------------------------------
# 10. .gitignore (only if absent — do NOT overwrite)
# ------------------------------------------------------------

if [ ! -f .gitignore ]; then
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual envs
.venv/
venv/
env/

# OS
.DS_Store
Thumbs.db

# Editors
.vscode/
.idea/
*.swp

# Jupyter
.ipynb_checkpoints/

# Backup files
*~
*.bak

# Generated outputs
outputs/
fig_test/
convergence_figures/
figures_crowd_judgement/
figures_jigsaw/
figures_measuring_hate_speech/
figures_sbic/
figures_synthetic/
figures_synthetic_compare_corr/
figures_synthetic_correlated/

# Internal benchmark outputs
faircrowd/benchmark_new/exp_save/

# Datasets / large files
data/
*.parquet
*.pkl
*.tgz
EOF
else
    echo "[10] .gitignore already exists, leaving it untouched."
fi

# ------------------------------------------------------------
# 11. Makefile (only if absent)
# ------------------------------------------------------------

if [ ! -f Makefile ]; then
# Use literal tabs for Make recipes — tabs MUST be tabs, not spaces.
cat > Makefile <<'EOF'
.PHONY: install synthetic synthetic-correlated compare mhs-diagnose mhs-target-race sbic clean

install:
	pip install -r requirements.txt
	pip install datasets pyarrow

synthetic:
	python scripts/run_experiments_synthetic.py

synthetic-correlated:
	python scripts/run_experiments_synthetic_correlated.py

compare:
	python scripts/run_compare_independent_vs_correlated.py --n-repet 10

mhs-diagnose:
	python scripts/run_experiments_measuring_hate_speech.py --diagnose-only

mhs-target-race:
	python scripts/run_experiments_measuring_hate_speech.py \
	    --s-col target_race \
	    --max-items 6000 --max-annotators 800 --n-repet 10 \
	    --eps-list 0.01 0.03 0.05 0.10 0.15 0.20 0.30

sbic:
	python scripts/run_experiments_sbic.py \
	    --max-items 5000 --max-annotators 600 --n-repet 10

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".DS_Store" -delete
	find . -name "*~" -delete
EOF
else
    echo "[11] Makefile already exists, leaving it untouched."
fi

# ------------------------------------------------------------
# 12. Init git if absent
# ------------------------------------------------------------

if [ ! -d .git ]; then
    git init
    echo "[12] Initialized git repository."
fi

# ------------------------------------------------------------
# 13. Pre-commit safety checks (do NOT auto-commit)
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "Pre-commit safety checks"
echo "============================================================"

echo ""
echo "Files larger than 50 MB (would block GitHub push):"
find . -type f -size +50M -not -path "./.git/*" -not -path "./.venv/*" 2>/dev/null || echo "  (none)"

echo ""
echo "Files larger than 10 MB (worth reviewing):"
find . -type f -size +10M -not -path "./.git/*" -not -path "./.venv/*" 2>/dev/null || echo "  (none)"

echo ""
echo "Tracked CSV/PDF/PNG outside outputs/ that should maybe be ignored:"
find . -type f \( -name "*.csv" -o -name "*.pdf" -o -name "*.png" \) \
    -not -path "./.git/*" \
    -not -path "./.venv/*" \
    -not -path "./outputs/*" \
    -not -path "./paper_figures/*" \
    -not -path "./faircrowd/datasets/*" \
    | head -20 || true

echo ""
echo "Current git status:"
git status --short
echo ""

# ------------------------------------------------------------
# 14. Final guidance — do NOT auto-commit
# ------------------------------------------------------------

cat <<EOF

============================================================
Reorganization done. NO commit was created.

Next steps (manual review recommended):

  1. Inspect the changes:
       git status
       git diff --stat

  2. Test that scripts still run:
       make mhs-diagnose

  3. If everything is clean:
       git add .
       git commit -m "Organize reproducible experiment repository"
       git branch -M main
       git remote add origin https://github.com/<your-username>/FairCrowd-ICML.git
       git push -u origin main

  If a remote already exists:
       git push -u origin main

============================================================
EOF