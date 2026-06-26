from ultralytics import YOLO
import argparse
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8-Seg.")
    parser.add_argument("--config", default="configs/gpu_quick.yaml", help="Training config yaml.")
    parser.add_argument("--device", default=None, help="Override config device, for example 0, 0,1, or cpu.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device

    model_name = cfg.pop("model")
    model = YOLO(model_name)
    model.train(**cfg)


if __name__ == "__main__":
    main()