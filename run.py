#!/usr/bin/env python
"""Run a PDDLVM experiment from a JSON config (no interaction needed).

Usage:
    python run.py configs/ve_prior_default.json [--steps N] [--seed S]
                                                [--out-dir DIR]

Artifacts (metrics.json, loss.png, samples.png, checkpoints) land in a
versioned directory under the configured run_dir.
"""

import argparse

from pddlvm.experiment import Experiment, ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a JSON experiment config")
    parser.add_argument("--steps", type=int, default=None,
                        help="override training steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="override random seed")
    parser.add_argument("--out-dir", default=None,
                        help="override run_dir")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_file(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.out_dir is not None:
        cfg.run_dir = args.out_dir

    run_dir, metrics = Experiment(cfg).run(steps=args.steps)
    print(f"run dir : {run_dir}")
    print("final   : " + ", ".join(
        f"{k}={v:.4f}" for k, v in metrics["final"].items()))


if __name__ == "__main__":
    main()
