#!/usr/bin/env python3
"""
TMR - Transformer Motion Retargeting
Central orchestrator for all TMR operations

This is the main entry point for the TMR project. It provides:
- Interactive menu system (default when run without arguments)
- Command-line interface for scripting
- Full pipeline orchestration
- Data management, training, evaluation, experiments
- SLURM job submission and tracking
- Repository management and validation

Usage:
    python tmr.py                    # Interactive mode
    python tmr.py --help             # Show help
    python tmr.py validate           # Validate environment
    python tmr.py status             # Show repository status
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli import run_interactive, print_info


def main():
    """Main entry point"""
    # If no arguments, run interactive mode
    if len(sys.argv) == 1:
        print_info("Starting TMR Interactive Mode...")
        print_info("For command-line usage, run: python tmr.py --help")
        print()
        run_interactive()
        return

    # For now, just show help for command-line mode
    print_info("TMR Command-Line Interface")
    print()
    print("Available commands:")
    print("  python tmr.py              # Interactive mode (recommended)")
    print("  python tmr.py --help       # Show this help")
    print()
    print("The interactive mode provides full access to all TMR operations:")
    print("  • Data management (preprocess, validate, statistics)")
    print("  • Training (MLM, TMR, Mixformer, SGN, PMR, DMR)")
    print("  • Evaluation (baselines, anonymization, comprehensive)")
    print("  • Experiments & Ablations")
    print("  • Pipeline orchestration (run everything)")
    print("  • Repository management")
    print("  • SLURM job management")
    print()
    print("For the best experience, run without arguments to enter interactive mode.")


if __name__ == "__main__":
    main()
