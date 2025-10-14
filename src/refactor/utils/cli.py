# utils/cli.py
import argparse
import yaml
from experiments.registry import EXPERIMENTS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment name (linear, vae, ...)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="Key=Value overrides")
    return parser.parse_args()

def load_config(path, overrides):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for o in overrides:
        k, v = o.split("=")
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        cfg[k] = v
    return cfg

