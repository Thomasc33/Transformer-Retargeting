Repository structure (target, simplified)

- data/
  - models/                 # migrated saved models (e.g., trained_models/*)
  - models_output/          # migrated training outputs (e.g., output/*)
  - ntu/, ntu120/, etri/    # processed datasets
  - nturgbd_raw/, etri_raw/ # raw datasets
- configs/                  # YAML/JSON config files
- docs/                     # EVALUATION.md, STRUCTURE.md, reports, guides
- eval/                     # unified evaluation package (python -m eval)
- logs/                     # logs and slurm logs (logs/slurm/*.out|*.err)
- src/                      # training and library code (main.py, data.py moved here progressively)

Transitional notes
- evaluation_suite/ has been migrated under eval/suite/ with a legacy shim (evaluation_suite/*) that re-exports from eval.suite.
- pmr.py is the single orchestrator for evaluation, dashboard, cleanup, and validation.
- results/ is ephemeral evaluation output; use pmr.py clean-results to reset.

Entry points (preferred)
- Interactive menu: `python pmr.py`
- Evaluate critical set: `python pmr.py eval --set critical`
- Evaluate one experiment: `python pmr.py eval --one baseline_comparison`
- Rebuild dashboard: `python pmr.py dash`
- Clean results: `python pmr.py clean-results`
- Validate env: `python pmr.py validate`

Dashboard and artifacts
- results.html is the single dashboard at repo root.
- Evaluation runs populate results/experiments/... and embed media.
- GIFs/images/videos are not gitignored.

Model paths
- Training outputs and legacy checkpoints are migrated to:
  - data/models/ (e.g., data/models/trained_models/...)
  - data/models_output/ (e.g., data/models_output/output/...)
- eval_model.py searches these paths automatically if legacy locations are missing.
