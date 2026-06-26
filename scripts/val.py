from ultralytics import YOLO
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a trained YOLOv8-Seg model.")
    parser.add_argument("--model", required=True, help="Path to best.pt.")
    parser.add_argument("--data", default="configs/coco_subset_seg.yaml", help="Dataset yaml.")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0", help="Device for validation, for example 0 or cpu.")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.val(data=args.data, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()