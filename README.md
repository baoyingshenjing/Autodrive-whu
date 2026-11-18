# 武汉大学自动驾驶技术课程成果

仓库统一保留 `final-report` 分支，集中保存四个阶段的最终成果及完整结课报告。

| 阶段 | 内容与复现说明 | 最终报告 |
|---|---|---|
| 实验一 | [nuScenes 多传感器处理与 YOLO](stage01/README.md) | [PDF](stage01/report/boston-seaport-yolo-report.pdf) · [LaTeX](stage01/report/boston-seaport-yolo-report.tex) |
| 实验二 | [PETR 多相机三维目标检测](stage02/README.md) | [PDF](stage02/report/petr-mini-report.pdf) · [LaTeX](stage02/report/petr-mini-report.tex) |
| 实验三 | [PointPillars LiDAR 三维目标检测](stage03/README.md) | [PDF](stage03/output/pdf/自动驾驶技术第三次实验报告_杜承敖_2024302112019.pdf) · [LaTeX](stage03/output/pdf/pointpillars-latex/main.tex) |
| 实验四 | [端到端行为克隆](stage04/README.md) | [PDF](stage04/output/pdf/自动驾驶技术第四次实验报告_杜承敖_2024302112019_增图版.pdf) · [LaTeX](stage04/output/pdf/report.tex) |
| 结课报告 | [系统组成、感知实践与规划决策综合分析](final-report/README.md) | [PDF](final-report/结课报告_目录修正版.pdf) · [LaTeX](final-report/结课报告_目录修正版.tex) |

## 文件范围

- 五份最终报告、对应 LaTeX 源码及全部编译必需配图。
- 四个阶段的实际运行源码、环境依赖说明、核心训练日志、配置、Loss 曲线与最终评估结果。
- YOLO 最佳模型、PETR 第 20 轮最终模型、PointPillars 公开基线与第 5 轮最终模型、行为克隆最佳模型。
- 数据集、完整 Python 环境、模拟器安装包、提交 ZIP、旧版报告、排版预览、重复控制台转录及中间轮次模型不纳入当前版本。

实验一的代码与记录沿用根目录 `source/`、`records/`；其报告、配图和最佳模型位于 `stage01/`。实验二至四分别位于 `stage02/`、`stage03/`、`stage04/`，结课报告位于 `final-report/`。

## 获取完整模型

PETR 最终模型约 388 MB，使用 Git LFS 保存；其余选定模型直接存于 Git。先安装 Git LFS，再克隆本仓库：

```bash
git lfs install
git clone --branch final-report https://github.com/baoyingshenjing/Autodrive-whu.git
cd Autodrive-whu
git lfs pull
```

[PETR 最终模型](stage02/PETR/outputs/run20/epoch_20.pth)需由 Git LFS 下载实际内容，普通源码 ZIP 不保证包含大文件本体。原始传感器数据和骨干初始化权重按各阶段复现说明从课程资料准备。

## 结果口径

YOLO、PETR 与 PointPillars 的传感器、任务、训练规模和评价指标不同，不能直接按 mAP 数值横向排名。实验四已完成离线验证和合成通信检查，尚无真实模拟器赛道闭环成绩；结课报告保留这些限制。
