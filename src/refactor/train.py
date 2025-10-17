# train.py
from utils.cli import parse_args, load_config
from utils.logging_utils import SimpleLogger
from experiments.registry import EXPERIMENTS

def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)

    # Get experiment name
    exp_name = cfg.get("experiment")
    if exp_name is None:
        raise ValueError("Config must include 'experiment' key, e.g. experiment: vae")

    # Get experiment class
    exp_class = EXPERIMENTS[exp_name]

    # Optional logger
    logger = SimpleLogger(save_dir=cfg.get("log_dir", "runs"))

    # Instantiate and run
    print(cfg, cfg.get("experiment"))
    experiment = exp_class(cfg, logger=logger)
    experiment.run()

if __name__ == "__main__":
    main()

