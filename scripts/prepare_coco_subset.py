from pathlib import Path
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict


DEFAULT_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "dog",
    "cat",
    "bottle",
    "chair",
    "laptop",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def polygon_area(points):
    area = 0.0
    for i in range(0, len(points), 2):
        x1, y1 = points[i], points[i + 1]
        x2, y2 = points[(i + 2) % len(points)], points[(i + 3) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def normalize_polygon(points, width, height):
    normalized = []
    for i in range(0, len(points), 2):
        x = min(max(points[i] / width, 0.0), 1.0)
        y = min(max(points[i + 1] / height, 0.0), 1.0)
        normalized.extend([x, y])
    return normalized


def best_polygon(segmentation):
    if not isinstance(segmentation, list):
        return None
    polygons = [p for p in segmentation if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
    if not polygons:
        return None
    return max(polygons, key=polygon_area)


def reset_yolo_dirs(output_root):
    for split in ["train", "val", "test"]:
        for kind in ["images", "labels"]:
            path = output_root / kind / split
            path.mkdir(parents=True, exist_ok=True)
            for file in path.iterdir():
                if file.is_file():
                    file.unlink()


def write_yaml(output_root, class_names):
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    text = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"names:\n{names}\n"
    )
    Path("configs/coco_subset_seg.yaml").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert a COCO instance segmentation subset to YOLO-Seg format.")
    parser.add_argument("--images", required=True, help="Folder containing COCO images, for example val2017.")
    parser.add_argument("--annotations", required=True, help="COCO instances json, for example instances_val2017.json.")
    parser.add_argument("--output", default="datasets/coco_subset_seg", help="YOLO dataset output folder.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="COCO category names to keep.")
    parser.add_argument("--max-per-class", type=int, default=200, help="Approximate maximum selected images per class.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_root = Path(args.images)
    annotation_path = Path(args.annotations)
    output_root = Path(args.output)

    if not image_root.exists():
        raise FileNotFoundError(f"Image folder not found: {image_root}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {cat["name"]: cat["id"] for cat in coco["categories"]}
    missing = [name for name in args.classes if name not in categories]
    if missing:
        raise ValueError(f"Classes not found in COCO annotations: {missing}")

    selected_cat_ids = {categories[name] for name in args.classes}
    cat_id_to_local = {categories[name]: i for i, name in enumerate(args.classes)}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}

    anns_by_image = defaultdict(list)
    present_cats_by_image = defaultdict(set)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cat_id = ann["category_id"]
        if cat_id not in selected_cat_ids:
            continue
        polygon = best_polygon(ann.get("segmentation"))
        if polygon is None:
            continue
        image_id = ann["image_id"]
        if image_id not in images:
            continue
        anns_by_image[image_id].append(ann)
        present_cats_by_image[image_id].add(cat_id)

    rng = random.Random(args.seed)
    candidate_ids = list(anns_by_image.keys())
    rng.shuffle(candidate_ids)

    selected_ids = []
    class_image_counts = Counter()
    for image_id in candidate_ids:
        cats = present_cats_by_image[image_id]
        if any(class_image_counts[cat_id] < args.max_per_class for cat_id in cats):
            selected_ids.append(image_id)
            for cat_id in cats:
                class_image_counts[cat_id] += 1
        if all(class_image_counts[cat_id] >= args.max_per_class for cat_id in selected_cat_ids):
            break

    rng.shuffle(selected_ids)
    train_end = int(len(selected_ids) * args.train_ratio)
    val_end = train_end + int(len(selected_ids) * args.val_ratio)
    split_ids = {
        "train": selected_ids[:train_end],
        "val": selected_ids[train_end:val_end],
        "test": selected_ids[val_end:],
    }

    reset_yolo_dirs(output_root)

    split_instance_counts = {split: Counter() for split in split_ids}
    for split, image_ids in split_ids.items():
        for image_id in image_ids:
            img = images[image_id]
            src = image_root / img["file_name"]
            if not src.exists() or src.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            dst_image = output_root / "images" / split / img["file_name"]
            shutil.copy2(src, dst_image)

            label_lines = []
            for ann in anns_by_image[image_id]:
                polygon = best_polygon(ann.get("segmentation"))
                if polygon is None:
                    continue
                normalized = normalize_polygon(polygon, img["width"], img["height"])
                cls = cat_id_to_local[ann["category_id"]]
                coords = " ".join(f"{value:.6f}" for value in normalized)
                label_lines.append(f"{cls} {coords}")
                split_instance_counts[split][ann["category_id"]] += 1

            label_path = output_root / "labels" / split / f"{Path(img['file_name']).stem}.txt"
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    total_instance_counts = Counter()
    for label_file in (output_root / "labels").glob("*/*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                total_instance_counts[int(parts[0])] += 1

    write_yaml(output_root.resolve(), args.classes)

    print(f"Output dataset: {output_root.resolve()}")
    print(f"Selected images: {len(selected_ids)}")
    for split, image_ids in split_ids.items():
        print(f"{split}: {len(image_ids)} images")
    print("\nInstance counts:")
    for class_id, name in enumerate(args.classes):
        print(f"{name:>12}: {total_instance_counts[class_id]}")
    print("\nUpdated configs/coco_subset_seg.yaml")


if __name__ == "__main__":
    main()
