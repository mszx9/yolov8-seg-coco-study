from ultralytics import YOLO
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8-Seg prediction on one image or a folder.")
    parser.add_argument("--model", required=True, help="Path to best.pt or a YOLO model name.")
    parser.add_argument("--source", required=True, help="Image, folder, or video path.")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, device="cpu", save=True)


if __name__ == "__main__":
    main()
