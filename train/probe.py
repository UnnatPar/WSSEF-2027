import argparse
import os
import sys

# See train/pretrain_mae.py for why this is needed: a directly-invoked
# script only gets its own directory on sys.path, not the repo root -- so
# even `from train.supervised import ...` (a sibling module in the same
# package as this file) fails without it, since `train` itself isn't
# importable as a package from inside its own directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.supervised import build_probe_model, run_supervised

__all__ = ["build_probe_model", "main"]


def main(config_path: str, fast_dev_run: bool = False):
    # Early stopping was originally off here on the assumption that probing a
    # frozen encoder is cheap -- it isn't: the forward pass still runs the
    # full 5.7M-param encoder every step, so a fixed 30-epoch schedule costs
    # real A100 hours (~59hr measured on this run) same as full finetuning.
    return run_supervised(config_path, fast_dev_run, early_stopping=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    main(args.config, fast_dev_run=args.fast_dev_run)
