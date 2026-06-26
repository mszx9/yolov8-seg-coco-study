# 基于 YOLOv8-Seg 的单图像多类目标分割与分类研究方案

## 1. 研究目标

本项目研究基于 YOLOv8-Seg 的单图像多类目标分割与分类方法。模型需要在一张输入图像中完成多类别目标识别、目标定位和实例掩膜分割。

核心对比实验：

| 编号 | 模型 | 图像增强 | 目的 |
|---|---|---|---|
| E1 | YOLOv8n-seg | 否 | 轻量模型基线 |
| E2 | YOLOv8n-seg | 是 | 分析增强对轻量模型的影响 |
| E3 | YOLOv8s-seg | 否 | 较大模型基线 |
| E4 | YOLOv8s-seg | 是 | 分析增强对较大模型的影响 |

## 2. CPU 实验策略

当前机器没有 NVIDIA 显卡，因此训练命令全部使用 `device=cpu`。CPU 训练速度较慢，建议采用三阶段路线：

1. 使用 `coco8-seg.yaml` 快速跑通环境。
2. 使用自建小规模多类数据集完成初步实验。
3. 在时间允许时增加 epoch、数据量和图像尺寸，完成正式实验。

推荐 CPU 参数：

| 阶段 | 模型 | imgsz | batch | epochs |
|---|---|---:|---:|---:|
| 快速验证 | YOLOv8n-seg | 416 | 2 | 3 |
| 正式 n 模型 | YOLOv8n-seg | 512 | 2 | 50 |
| 正式 s 模型 | YOLOv8s-seg | 512 | 1 | 50 |

## 3. 数据集设计

推荐类别：

| 类别编号 | 类别名 |
|---:|---|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 4 | bus |
| 5 | dog |
| 6 | cat |
| 7 | bottle |
| 8 | chair |
| 9 | laptop |

数据目录：

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

YOLO-Seg 标签格式：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

其中坐标均为 0 到 1 之间的归一化多边形坐标。

## 4. 训练命令

先跑通环境：

```bash
python scripts/train.py --config configs/cpu_quick.yaml
```

正式实验：

```bash
python scripts/train.py --config configs/cpu_formal_yolov8n_baseline.yaml
python scripts/train.py --config configs/cpu_formal_yolov8n_aug.yaml
python scripts/train.py --config configs/cpu_formal_yolov8s_baseline.yaml
python scripts/train.py --config configs/cpu_formal_yolov8s_aug.yaml
```

验证模型：

```bash
python scripts/val.py --model runs/segment/yolov8n_cpu_baseline/weights/best.pt
```

预测单张图像：

```bash
python scripts/predict.py --model runs/segment/yolov8n_cpu_baseline/weights/best.pt --source test.jpg
```

## 5. 实验记录表

| 实验编号 | 模型 | 图像增强 | Precision | Recall | mAP50-Box | mAP50-Mask | mAP50-95-Mask | FPS | 参数量 | 模型大小 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | YOLOv8n-seg | 否 |  |  |  |  |  |  |  |  |
| E2 | YOLOv8n-seg | 是 |  |  |  |  |  |  |  |  |
| E3 | YOLOv8s-seg | 否 |  |  |  |  |  |  |  |  |
| E4 | YOLOv8s-seg | 是 |  |  |  |  |  |  |  |  |

消融实验：

| 实验 | Mosaic | MixUp | HSV | Flip | mAP50-Mask | mAP50-95-Mask |
|---|---|---|---|---|---:|---:|
| A0 | 否 | 否 | 否 | 否 |  |  |
| A1 | 是 | 否 | 否 | 否 |  |  |
| A2 | 是 | 是 | 否 | 否 |  |  |
| A3 | 是 | 是 | 是 | 是 |  |  |

## 6. 创新点

1. 面向单图像多类目标场景，构建 YOLOv8-Seg 实例分割与分类实验流程。
2. 设计 Mosaic、MixUp、HSV 色彩扰动、随机翻转、尺度变换组合增强策略。
3. 对 YOLOv8n-seg 和 YOLOv8s-seg 进行精度、速度、参数量和模型大小对比。
4. 建立包含 Box 指标、Mask 指标和推理效率的综合评价体系。

## 7. 论文目录

```text
摘要
Abstract

第1章 绪论
第2章 相关理论与技术基础
第3章 数据集构建与实验环境
第4章 基于 YOLOv8-Seg 的多类目标分割方法
第5章 实验结果与分析
第6章 总结与展望

参考文献
致谢
附录
```
