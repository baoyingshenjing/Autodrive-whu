# 自动驾驶技术实验一：nuScenes 多传感器处理与 YOLO

姓名：杜承敖；学号：2024302112019。本目录补全最终报告、LaTeX 配图及最佳模型；实际运行源码和训练记录沿用仓库根目录的 `source/`、`records/`，不重复复制。

## 成果入口

- [最终报告 PDF](report/boston-seaport-yolo-report.pdf)、[LaTeX 源码](report/boston-seaport-yolo-report.tex)及[必需配图](report/assets/)
- [YOLO11n 最佳模型](checkpoints/best.pt)
- [数据读取与投影源码](../source/scripts/)、[YOLO 训练评估源码](../source/yolo_workflow/)
- [训练配置](../records/args.yaml)、[20 轮记录](../records/results.csv)、[独立 test 指标](../records/boston_test_summary.json)和[预测展示记录](../records/boston_test_predictions_summary.json)
- [课程实验要求](../docs/自动驾驶课程实践---数据集+YOLO.pdf)

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.511 | 0.394 | 0.403 | 0.217 |

Boston Seaport 数据划分为训练 9,420 帧、验证 1,402 帧、独立测试 2,709 帧，10 类。YOLO11n 从公开预训练模型开始，输入 640、batch=8、训练 20 轮。devkit 实践包含 Camera/LiDAR 读取、坐标变换及投影；材料缺少 Radar 原始 PCD，未产生 Radar 实测叠加结果。

## 复现

环境为 Python 3.10、PyTorch 2.8.0+cu126、Ultralytics 8.4.152。以下命令均在仓库根目录运行：

```bash
pip install -r source/requirements-devkit.txt
pip install -r source/yolo_workflow/requirements-yolo.txt
python source/scripts/build_devkit_evidence.py --dataroot <NUSCENES_ROOT> --asset-root <ASSET_ROOT> --output-dir runs/devkit
python source/yolo_workflow/train.py --model yolo11n.pt --dataset-root <DATASET_ROOT> --imgsz 640 --batch 8 --epochs 20 --device 0 --workers 4 --project runs/checkpoints --name boston_20ep
python source/yolo_workflow/evaluate.py --model stage01/checkpoints/best.pt --dataset-root <DATASET_ROOT> --splits test --imgsz 640 --batch 8 --device 0 --workers 4 --project runs/evaluation --name boston_test
python source/yolo_workflow/visualize.py --model stage01/checkpoints/best.pt --dataset-root <DATASET_ROOT> --split test --max-images 12 --imgsz 640 --device 0 --workers 4 --project runs/visualization --name boston_test
python source/scripts/build_prediction_montage.py runs/visualization/boston_test runs/visualization/boston_test_montage.jpg
```

`<DATASET_ROOT>` 应包含 `data.yaml` 及 `images/{train,val,test}`、`labels/{train,val,test}`。原始数据另行准备。模型和记录为实际保存结果，20 轮训练并不代表充分调参或最优性能。

## 编译报告

在 `stage01/report/` 连续执行三次 `xelatex -interaction=nonstopmode -halt-on-error boston-seaport-yolo-report.tex`。源码仅将配图路径整理为本目录的 `assets/`，报告正文和最终 PDF 保持原样。
