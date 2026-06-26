from ultralytics import YOLO
import argparse
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8-Seg on CPU-friendly settings.")
    parser.add_argument("--config", default="configs/cpu_quick.yaml", help="Training config yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg.pop("model")
    model = YOLO(model_name)
    model.train(**cfg)


if __name__ == "__main__":
    main()
