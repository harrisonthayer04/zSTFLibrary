#!/usr/bin/env python
"""Train multiple models (single-run or search) in one command.

Usage examples:

# Train single-run configs for models 1 and 4
python train_multi.py --models 1,4

# Train hyper-parameter search configs for all models
python train_multi.py --models all --search
"""
import argparse
import subprocess
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"
SINGLE_TPL = "model_{idx}_config.yaml"
SEARCH_TPL = "model_{idx}_search.yaml"
ALL_MODELS = ["1", "2", "3", "4", "5"]


def _run(cmd: list):
    print("=" * 60)
    print("Running:", " ".join(cmd))
    print("=" * 60)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Train one or more models in sequence.")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model numbers to train (e.g. 1,3,5) or 'all'",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Use *_search.yaml configs instead of single-run configs.",
    )
    args = parser.parse_args()

    if args.models.lower() == "all":
        indices = ALL_MODELS
    else:
        indices = [m.strip() for m in args.models.split(",") if m.strip() in ALL_MODELS]
        if not indices:
            raise ValueError("No valid model indices provided. Valid choices: 1-5 or 'all'.")

    tpl = SEARCH_TPL if args.search else SINGLE_TPL

    for idx in indices:
        cfg_file = CONFIG_DIR / tpl.format(idx=idx)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_file}")
        _run([
            "python",
            str(Path(__file__).parent / "run_experiment.py"),
            "--config",
            str(cfg_file),
        ])

    print("All requested trainings complete.")


if __name__ == "__main__":
    main() 