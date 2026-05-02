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
