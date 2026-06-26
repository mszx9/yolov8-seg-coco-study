# COCO 子集数据集准备指南

本项目建议使用 COCO 2017 `val2017` 作为稳定起步数据源。它比完整 `train2017` 小很多，更适合 CPU 环境，同时仍然包含多类、多目标、遮挡和尺度变化。

## 1. 需要下载的文件

下载并解压到 `D:/tuxiang/downloads/coco2017`：

```text
D:/tuxiang/downloads/coco2017/
├── val2017/
│   ├── 000000000139.jpg
│   └── ...
└── annotations/
    └── instances_val2017.json
```

常用来源：

```text
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

## 2. 推荐类别

```text
person, bicycle, car, motorcycle, bus, dog, cat, bottle, chair, laptop
```

这些类别常见、样本量相对充足，也适合论文中讨论多类目标分割与分类。

## 3. 转换为 YOLOv8-Seg 格式

在已经激活 `yolov8seg-cpu` 环境的终端中运行：

```bash
python scripts/prepare_coco_subset.py --images D:/tuxiang/downloads/coco2017/val2017 --annotations D:/tuxiang/downloads/coco2017/annotations/instances_val2017.json --max-per-class 200
```

转换后会生成：

```text
datasets/coco_subset_seg/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

同时会更新：

```text
configs/coco_subset_seg.yaml
```

## 4. 检查数据集

```bash
python danmu.py check-data
```

如果 train、val、test 都有图片和标签，就可以开始正式训练。

## 5. CPU 训练建议

先运行 YOLOv8n-seg 基线：

```bash
python danmu.py train-n
```

再运行 YOLOv8n-seg 增强实验：

```bash
python danmu.py train-n-aug
```

确认流程和结果稳定后，再运行 YOLOv8s-seg：

```bash
python danmu.py train-s
python danmu.py train-s-aug
```

CPU 训练速度较慢，如果时间紧，优先完成 YOLOv8n-seg 的基线与增强对比。
