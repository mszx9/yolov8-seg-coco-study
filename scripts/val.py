from ultralytics import YOLO
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a trained YOLOv8-Seg model on CPU.")
    parser.add_argument("--model", required=True, help="Path to best.pt.")
    parser.add_argument("--data", default="configs/coco_subset_seg.yaml", help="Dataset yaml.")
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.val(data=args.data, imgsz=args.imgsz, device="cpu")


if __name__ == "__main__":
    main()
