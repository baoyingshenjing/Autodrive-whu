# Boston Seaport 多传感器数据处理与 YOLO 目标检测

本仓库整理了自动驾驶技术课程中 Boston Seaport 场景的实验代码与脱敏结果记录。实验包含 nuScenes 元数据读取、Camera 与 LiDAR 数据显示、三维点投影到二维图像、YOLO 数据组织、YOLO11n 训练、独立 test 集评估及预测可视化。

[课程实验要求 PDF](docs/自动驾驶课程实践---数据集+YOLO.pdf)

## 仓库结构

```text
docs/                 课程实验要求
source/scripts/       devkit 证据图与预测拼图脚本
source/yolo_workflow/ YOLO 训练、评估和可视化脚本
records/              训练参数、20 轮日志和独立 test 结果
```

数据集、模型权重和完整训练输出体积较大，未包含在仓库中。`data/boston-seaport/` 表示本地转换后的 YOLO 数据目录；`runs/` 表示训练与评估输出目录。

## 环境

- Python 3.10
- PyTorch 2.8.0 + CUDA 12.6
- Ultralytics 8.4.152

安装依赖：

```bash
pip install -r source/requirements-devkit.txt
pip install -r source/yolo_workflow/requirements-yolo.txt
```

## 数据准备

YOLO 数据目录应包含 `data.yaml`，并按 `images/{train,val,test}` 与 `labels/{train,val,test}` 组织。当前实验使用 10 个类别，共有训练集 9,420 帧、验证集 1,402 帧、测试集 2,709 帧。

本次课程材料中可用的原始传感器文件覆盖前视相机与顶部 LiDAR，因此 devkit 证据脚本完成了 Camera/LiDAR 的读取、坐标变换和叠加显示。元数据仍包含 Radar 通道定义，但缺少对应 Radar 原始 PCD 文件，未生成 Radar 实测叠加结果。

## 运行方法

以下命令在仓库根目录执行，其中 `<DATASET_ROOT>` 是包含 `data.yaml` 的本地数据目录，`<NUSCENES_ROOT>` 与 `<ASSET_ROOT>` 分别是 nuScenes 元数据目录和传感器文件目录。

生成 devkit 数据读取及投影证据：

```bash
python source/scripts/build_devkit_evidence.py --dataroot <NUSCENES_ROOT> --asset-root <ASSET_ROOT> --output-dir runs/devkit
```

从预训练 YOLO11n 开始训练 20 轮：

```bash
python source/yolo_workflow/train.py --model weights/yolo11n.pt --dataset-root <DATASET_ROOT> --imgsz 640 --batch 8 --epochs 20 --device 0 --workers 4 --project runs/checkpoints --name boston_20ep
```

使用最优权重对独立 test 集评估：

```bash
python source/yolo_workflow/evaluate.py --model runs/checkpoints/boston_20ep/weights/best.pt --dataset-root <DATASET_ROOT> --splits test --imgsz 640 --batch 8 --device 0 --workers 4 --project runs/evaluation --name boston_test
```

抽取 12 张 test 图像生成预测结果：

```bash
python source/yolo_workflow/visualize.py --model runs/checkpoints/boston_20ep/weights/best.pt --dataset-root <DATASET_ROOT> --split test --max-images 12 --imgsz 640 --device 0 --workers 4 --project runs/visualization --name boston_test
```

生成预测拼图：

```bash
python source/scripts/build_prediction_montage.py runs/visualization/boston_test runs/visualization/boston_test_montage.jpg
```

## 实验结果

模型为预训练 YOLO11n，输入尺寸为 640，batch size 为 8，训练 20 轮。使用 `best.pt` 在独立 test 集上的结果为：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.511 | 0.394 | 0.403 | 0.217 |

详细训练参数与逐轮日志见 [`records/args.yaml`](records/args.yaml) 和 [`records/results.csv`](records/results.csv)，test 评估与 12 张预测记录见 `records/` 中的两个 JSON 文件。

20 轮训练满足课程实验的最低训练要求，但不代表模型已经过充分调参或达到最优性能。
