"""
Compatibility wrapper for eval_model_main

The canonical evaluation implementation is now at src/evaluation/eval_model_main.py.
This module re-exports all functions for backward compatibility.
"""
from pathlib import Path
import sys

# Import all functions from eval_model_main
try:
    # Import the main module
    from src.evaluation import eval_model_main as _eval_model_main

    # Re-export all public functions
    from src.evaluation.eval_model_main import *  # noqa: F401,F403

    # Provide main() function for CLI
    def main():
        """Entry point for command-line execution."""
        if hasattr(_eval_model_main, 'main'):
            return _eval_model_main.main()
        else:
            print("Error: main() function not found in eval_model_main.py")
            return 1

except ImportError as e:
    print(f"Error importing eval_model_main: {e}")
    print("Make sure eval_model_main.py exists at src/evaluation/")
    raise

