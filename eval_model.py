"""
Compatibility shim: evaluation has moved to src/evaluation/eval_model.py
This module re-exports the public API to avoid breaking legacy imports.
"""
from src.evaluation.eval_model import *  # noqa: F401,F403

if __name__ == "__main__":
    # Delegate CLI behavior to the consolidated module
    try:
        from src.evaluation.eval_model import main as _main
    except Exception:
        # Fallback: attempt to find a callable entrypoint
        import sys
        sys.exit("src.evaluation.eval_model.main not found")
    _main()
