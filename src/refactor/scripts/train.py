# train.py
from utils.cli import parse_args, load_config
from experiments.registry import EXPERIMENTS

def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    exp_class = EXPERIMENTS[cfg.get("experiment")]
    experiment = exp_class(cfg)
    experiment.run()

if __name__ == "__main__":
    main()

