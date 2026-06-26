from pathlib import Path
import argparse
import yaml
from collections import Counter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_files(path: Path, suffixes=None) -> int:
    if not path.exists():
        return 0
    if suffixes is None:
        return sum(1 for p in path.iterdir() if p.is_file())
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def load_names(data_yaml: Path):
    if not data_yaml.exists():
        return {}
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def count_instances(root: Path):
    counts = Counter()
    for label_file in (root / "labels").glob("*/*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a YOLO segmentation dataset folder.")
    parser.add_argument("--root", default="datasets/coco_subset_seg", help="Dataset root folder.")
    parser.add_argument("--data", default="configs/coco_subset_seg.yaml", help="Dataset yaml for class names.")
    args = parser.parse_args()

    root = Path(args.root)
    names = load_names(Path(args.data))
    splits = ["train", "val", "test"]
    print(f"Dataset root: {root.resolve()}")

    ok = True
    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_count = count_files(image_dir, IMAGE_SUFFIXES)
        label_count = count_files(label_dir, {".txt"})
        print(f"{split:>5}: images={image_count:>5} labels={label_count:>5}")
        if image_count == 0 and split in {"train", "val"}:
            ok = False
        if image_count != label_count:
            print(f"      warning: image/label counts differ in {split}")

    if not ok:
        print("Status: dataset is not ready yet. Add images and YOLO-Seg polygon labels first.")
    else:
        counts = count_instances(root)
        if counts:
            print("\nInstance counts:")
            for class_id in sorted(names or counts):
                name = names.get(class_id, str(class_id))
                print(f"{class_id:>2} {name:<12} {counts[class_id]:>6}")
        print("Status: basic folder check passed.")


if __name__ == "__main__":
    main()
