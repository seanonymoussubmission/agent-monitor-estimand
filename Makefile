.PHONY: test reproduce-main
test:
	python3 tests/test_metrics.py

reproduce-main: test
	python3 scripts/62_pair_decomposition.py --probe FINAL/merged.npz --k 1 --layer 20 --boot 2000
	python3 scripts/24_lph_analysis.py --probe FINAL/merged.npz --out FINAL/analysis.json --boot 2000 --perm 2000
	python3 scripts/63_automata_reeval.py --shards 12
	python3 scripts/93_factorial_and_taskboot.py
	python3 scripts/77_direct_leakage.py --boot 600
	python3 scripts/79_abort_vs_allocate.py --trials 150
	python3 scripts/90_predict_inflation.py
