"""Command-line entry point for the local planner/executor workflow."""
import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd

from core.execution_engine import ExecutionEngine


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a planned data analysis task")
    parser.add_argument("dataset", help="Path to a CSV dataset")
    parser.add_argument("--goal", required=True, help="Analysis goal in plain language")
    parser.add_argument("--target", help="Optional target column for model analysis")
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.dataset)
    goal = args.goal + (f" Target: {args.target}" if args.target else "")
    result = ExecutionEngine(max_retries=args.max_retries).run(df, goal)
    print(f"Task {result['task_id']}: {result['status']}")
    print(f"Plan: {result['state']['plan']}")
    print(f"Critic: {result['state']['critic_result']}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
