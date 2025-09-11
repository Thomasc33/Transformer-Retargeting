# Cross-platform Makefile (macOS/Linux/WSL/MinGW). Use 'python3' override if needed.

PY?=python
PIP?=pip

# Basic checks
.PHONY: help
help:
	@echo "Targets:"
	@echo "  make eval-all           # Submit all evaluations and refresh dashboard"
	@echo "  make eval-critical      # Submit critical set"
	@echo "  make eval-one EXP=...   # Run a single experiment locally"
	@echo "  make dash               # Regenerate results.html"
	@echo "  make clean-results      # Delete evaluation outputs (keeps models)"
	@echo "  make validate           # Quick env checks (torch/CUDA/data)"
	@echo "  make status             # List submitted Slurm jobs (if any)"
	@echo "  make logs               # Tail latest slurm logs"

.PHONY: eval-all
eval-all:
	$(PY) tmr.py eval --set complete

.PHONY: eval-critical
eval-critical:
	$(PY) tmr.py eval --set critical

.PHONY: eval-one
eval-one:
	@if [ -z "$(EXP)" ]; then echo "Usage: make eval-one EXP=experiment_name"; exit 1; fi
	$(PY) - <<'PY'
import sys, subprocess
subprocess.check_call([sys.executable, 'tmr.py', 'eval', '--one', "$(EXP)"])
PY

.PHONY: dash
dash:
	$(PY) tmr.py dash

.PHONY: clean-results
clean-results:
	$(PY) tmr.py clean-results

.PHONY: validate
validate:
	$(PY) tmr.py validate

.PHONY: status
status:
	@if command -v squeue >/dev/null 2>&1; then squeue -u $$USER || true; else echo "squeue not available"; fi

.PHONY: logs
logs:
	@if ls logs/slurm/*.out >/dev/null 2>&1; then tail -n 100 logs/slurm/*.out; else echo "No slurm logs yet"; fi

