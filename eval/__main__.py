"""python -m eval entrypoint.

Examples:
  python -m eval --set critical     # run critical set
  python -m eval --one baseline     # run single experiment
  python -m eval --dash             # rebuild dashboard
"""
import argparse
import sys
from . import run_all, run_one, dashboard


def main(argv=None):
    p = argparse.ArgumentParser(description="Unified evaluation entrypoint (python -m eval)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--set', choices=['critical','complete','quick','paper_ready'], help='Experiment set to run')
    g.add_argument('--one', help='Run a single experiment by name')
    g.add_argument('--dash', action='store_true', help='Regenerate dashboard')
    p.add_argument('--local', action='store_true')
    args = p.parse_args(argv)

    if args.dash:
        return dashboard()
    if args.one:
        return run_one(args.one)
    return run_all(args.set, args.local)


if __name__ == '__main__':
    main()

