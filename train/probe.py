import argparse

from train.supervised import build_probe_model, run_supervised

__all__ = ["build_probe_model", "main"]


def main(config_path: str, fast_dev_run: bool = False):
    return run_supervised(config_path, fast_dev_run, early_stopping=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    main(args.config, fast_dev_run=args.fast_dev_run)
