"""
YOLOv8-Seg GPU project entry point.

Run examples:
    python danmu.py quick
    python danmu.py check-data
    python danmu.py check-gpu
    python danmu.py train-n
"""

import argparse
import subprocess
import sys


COMMANDS = {
    "quick": ["scripts/train.py", "--config", "configs/gpu_quick.yaml"],
    "check-data": ["scripts/check_dataset.py"],
    "check-gpu": ["scripts/check_gpu.py"],
    "train-n": ["scripts/train.py", "--config", "configs/gpu_formal_yolov8n_baseline.yaml"],
    "train-n-aug": ["scripts/train.py", "--config", "configs/gpu_formal_yolov8n_aug.yaml"],
    "train-s": ["scripts/train.py", "--config", "configs/gpu_formal_yolov8s_baseline.yaml"],
    "train-s-aug": ["scripts/train.py", "--config", "configs/gpu_formal_yolov8s_aug.yaml"],
    "cpu-quick": ["scripts/train.py", "--config", "configs/cpu_quick.yaml"],
    "cpu-train-n": ["scripts/train.py", "--config", "configs/cpu_formal_yolov8n_baseline.yaml"],
    "cpu-train-n-aug": ["scripts/train.py", "--config", "configs/cpu_formal_yolov8n_aug.yaml"],
    "cpu-train-s": ["scripts/train.py", "--config", "configs/cpu_formal_yolov8s_baseline.yaml"],
    "cpu-train-s-aug": ["scripts/train.py", "--config", "configs/cpu_formal_yolov8s_aug.yaml"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8-Seg GPU project launcher.")
    parser.add_argument("command", choices=COMMANDS.keys())
    args = parser.parse_args()

    subprocess.check_call([sys.executable, *COMMANDS[args.command]])


if __name__ == "__main__":
    main()