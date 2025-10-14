# utils/logging_utils.py
import os, json

class SimpleLogger:
    def __init__(self, save_dir="runs"):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, "results.json")
        self.data = []

    def log(self, epoch, train_loss, val_loss):
        self.data.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

