"""Unified evaluation package.

Provides stable entry points while we migrate from evaluation_suite/ to eval/.
"""
from typing import Optional


def run_all(set_name: str = "critical", local: bool = False) -> None:
    from eval.suite.run_all_evaluations import main as _main
    import sys
    argv = [sys.argv[0], "--set", set_name]
    if local:
        argv += ["--local"]
    _main(argv)


def run_one(experiment: str) -> None:
    from eval.suite.run_single_experiment import main as _main
    import sys
    argv = [sys.argv[0], "--experiment", experiment]
    _main(argv)


def dashboard() -> None:
    from eval.suite.generate_dashboard import main as _main
    _main()

