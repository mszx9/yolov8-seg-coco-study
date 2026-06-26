# YOLOv8-Seg CPU Project

本项目用于完成“基于 YOLOv8-Seg 的单图像多类目标分割与分类研究”。

## 1. 创建 Conda 环境

```bash
conda env create -f environment-cpu.yml
conda activate yolov8seg-cpu
```

如果上面的环境创建失败，也可以手动创建：

```bash
conda create -n yolov8seg-cpu python=3.8.18
conda activate yolov8seg-cpu
pip install -r requirements-cpu.txt
```

## 2. 检查环境

```bash
yolo checks
```

建议先把 Ultralytics 的默认路径设置到当前项目，避免数据集或训练结果保存到旧目录：

```bash
yolo settings datasets_dir=D:/tuxiang/datasets runs_dir=D:/tuxiang/runs weights_dir=D:/tuxiang/weights
```

## 3. 先跑通 CPU 快速训练

```bash
python scripts/train.py --config configs/cpu_quick.yaml
```

该命令使用 Ultralytics 自带的 `coco8-seg.yaml` 小数据集，主要用于验证流程。

## 4. 准备正式数据集

推荐使用 COCO 子集。详细步骤见 `docs/coco_subset_guide.md`。

将图像放入：

```text
datasets/coco_subset_seg/images/train
datasets/coco_subset_seg/images/val
datasets/coco_subset_seg/images/test
```

将 YOLO-Seg 标签放入：

```text
datasets/coco_subset_seg/labels/train
datasets/coco_subset_seg/labels/val
datasets/coco_subset_seg/labels/test
```

检查数据集：

```bash
python scripts/check_dataset.py
```

## 5. 正式实验

```bash
python scripts/train.py --config configs/cpu_formal_yolov8n_baseline.yaml
python scripts/train.py --config configs/cpu_formal_yolov8n_aug.yaml
python scripts/train.py --config configs/cpu_formal_yolov8s_baseline.yaml
python scripts/train.py --config configs/cpu_formal_yolov8s_aug.yaml
```

CPU 训练 YOLOv8s-seg 会明显慢于 YOLOv8n-seg。建议先完成 YOLOv8n-seg 的基线和增强实验。

## 6. 查看完整方案

详见 `docs/project_plan.md`。
