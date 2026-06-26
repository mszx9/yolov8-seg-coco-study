"""
YOLOv8-Seg CPU project entry point.

Run examples:
    python danmu.py quick
    python danmu.py check-data
    python danmu.py train-n
"""

import argparse
import subprocess
import sys


COMMANDS = {
    "quick": ["scripts/train.py", "--config", "configs/cpu_quick.yaml"],
    "check-data": ["scripts/check_dataset.py"],
    "train-n": ["scripts/train.py", "--config", "configs/cpu_formal_yolov8n_baseline.yaml"],
    "train-n-aug": ["scripts/train.py", "--config", "configs/cpu_formal_yolov8n_aug.yaml"],
    "train-s": ["scripts/train.py", "--config", "configs/cpu_formal_yolov8s_baseline.yaml"],
    "train-s-aug": ["scripts/train.py", "--config", "configs/cpu_formal_yolov8s_aug.yaml"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8-Seg CPU project launcher.")
    parser.add_argument("command", choices=COMMANDS.keys())
    args = parser.parse_args()

    subprocess.check_call([sys.executable, *COMMANDS[args.command]])


if __name__ == "__main__":
    main()
