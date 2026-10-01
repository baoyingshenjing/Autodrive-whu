# 自动驾驶技术第二次实验：PETR 源码与复现说明

姓名：杜承敖  学号：2024302112019
专业：计算机科学与技术（雷军班）  课程：自动驾驶技术
学校：武汉大学  实验日期：2026-09-29

## 1. 仓库材料

- `PETR/`：预处理、训练、测试入口，PETR 插件、教学配置、框架源码、环境依赖及统一坐标显示脚本，共 597 个源码文件。
- `records/`：本次真实训练日志、逐轮 Loss 汇总与曲线、实际运行配置、81 帧评估指标及三帧展示记录。
- `README.md`：本复现说明。实验一内容保留在仓库原有目录。

本分支不上传报告 PDF、提交包 ZIP、报告 Markdown/TeX、排版资源或编译产物，也不上传原始数据、模型权重、Conda 环境、缓存及大型逐框预测文件。数据和骨干预训练权重从课程资料另行准备。代码保留原始许可证，PETR 与 mmdetection3d 的 LICENSE 位于对应源码目录。

## 2. 目录关系

```text
PETR/
  preprocess.py
  train.py
  test.py
  environment/
    requirements-win.txt
    requirements-lock.txt
    check_environment.py
  PETR-official/
    projects/                   PETR 插件和配置
    tools/                      框架训练和测试工具
    mmdetection3d/               本地 MMDetection3D 1.0.0rc6 源码
    ckpts/                      另行放置骨干预训练权重
  v1.0-mini/                    另行放置数据
    samples/CAM_*/              六路相机 JPG 文件
    v1.0-mini/                  nuScenes JSON 元数据
  outputs/setup/
    show_unified_cameras.py      六路相机统一坐标显示脚本
```

保留上述层次，不要仅把三个入口单独复制到其他位置。本课程使用 sys.path 导入附带 mmdetection3d 源码，不需要将它替换成其他版本或执行 pip install -e。

## 3. 运行环境

实际环境：Windows，Python 3.8.19，PyTorch 1.13.1+cu117，torchvision 0.14.1+cu117，mmcv-full 1.7.0，mmdet 2.28.2，mmsegmentation 0.30.0，CUDA runtime 11.7；本次 GPU 为 RTX 4060 Laptop，约 8 GB 显存。依赖清单及完整版本约束附在 environment/。

克隆仓库并切换到 `experiment-2` 分支后，从 `stage02/PETR` 目录执行，例如：

```bat
conda create -n petr-mini python=3.8.19
conda activate petr-mini
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
python -m pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
python -m pip install -r environment/requirements-win.txt -c environment/requirements-lock.txt
```

若 Python 3.8 下载时发生 HTTPS 握手错误，可用正常联网的工具从同一官方源缓存 Python3.8/Windows 对应 wheel 后离线安装。本次采用了这一方法，不使用不同版本替代依赖。训练前确认 CUDA、PETR 插件及 DCNv2 CUDA 运算正常；MMCV 算子错误时先核对 Torch/CUDA/MMCV 搭配。附带 environment/check_environment.py 为可选检查，教程流程不要求每次单独运行它。

## 4. 数据和骨干权重

将课程提供的完整 nuScenes mini 相机数据放入 PETR/v1.0-mini/。其中元数据路径必须是 PETR/v1.0-mini/v1.0-mini/sample_annotation.json，六路关键帧图像位于 samples/。本次共 10 场景、404 关键帧和 2424 张图片，不读取 LiDAR 点云或 sweeps。

将课程提供的 ResNet50 预训练文件 resnet50_msra-5891d200.pth 放到 PETR/PETR-official/ckpts/。该文件是骨干初始化权重，不是本实验训练出的完整 PETR 检测权重。

预处理会重新生成索引及当前运行位置的图片路径，不能复制原电脑的绝对路径索引代替这一过程。

## 5. 复现命令

以下命令均在 PETR 目录中运行，结果目录第一次运行时应为空：

```bat
conda activate petr-mini
python preprocess.py
python train.py --epochs 20 --work-dir outputs/run20
python test.py --checkpoint outputs/run20/epoch_20.pth
python outputs/setup/show_unified_cameras.py
```

预处理校验 323 帧训练/81 帧验证、场景与样本隔离及每帧六张图片。训练采用六路 192×512、batch=1、FP32、100 查询、32 深度采样、3 层解码器、种子42。

测试默认评估全部 81 帧，并将前三帧按阈值0.1、全帧共享 top-20 生成相机/俯视图及 visualization_info.json。绘图参数不改变评估结果。统一坐标脚本在完成正常 run20 后运行，把实际图像放在8米虚拟深度平面上展示标定关系，不重建真实场景深度。

### 断点恢复

如中断，从最近完整的 epoch_N.pth 恢复，必须换用新的结果目录，例如：

```bat
python train.py --resume-from outputs/run20/epoch_12.pth --epochs 20 --work-dir outputs/run20_resume
python test.py --checkpoint outputs/run20_resume/epoch_20.pth
```

--epochs 20 表示目标总轮数，不能理解为额外训练20轮。FINAL_RUN_DIR 为实际完成第20轮的目录：正常完成时为 outputs/run20，恢复完成时为新目录。统一坐标辅助脚本默认将图写入 outputs/run20；若需要放入恢复目录，可调整脚本中的 run 路径。

## 6. 本次实际结果与文件说明

本次未中断，FINAL_RUN_DIR=outputs/run20。第20轮完成6460迭代，最终权重可加载，已记录 loss 和权重均为有限值。训练曲线的轮平均值由每轮32个日志记录点计算，不是全部323批次的精确均值。

- mAP=0.0005689761970509904
- NDS=0.028768130622055653
- mATE=1.0263230303744824，mASE=0.9268084369992124
- mAOE=1.053263089268238，mAVE=0.8978921663626155，mAAE=0.8904629714028705

检测质量很差，实际图中框错位且预测类别高度集中于汽车。不能仅凭 loss 下降判断模型可靠。本次为 mini_val 验证评估，没有独立带标注 test 成绩。

复现后关键输出：epoch_20.pth、训练 .log/.log.json、test/predictions.pkl、test/eval/pts_bbox/results_nusc.json 与 metrics_summary.json，以及 test/visualizations/start0_count3_thr0.1_top20/。六相机可视化以绿色显示真实框、红色显示预测框；俯视图 x向前/y向左、单位米，蓝点为自车原点。

### 已上传的真实记录

`records/20260929_153058.log` 和 `.log.json` 保留原始训练输出；`epoch_loss_summary.json` 包含全部 20 轮的日志点均值，`loss_curve.png` 为实际 Loss 曲线。`petr_mini_r50.py` 为该次运行保存的配置。

`records/eval/metrics_summary.json` 和 `metrics_details.json` 为官方评估输出，`prediction_stats.json` 汇总 81 帧、6783 个预测框，`experiment_validation.json` 记录划分、轮数及有限值检查结果。`records/visualization/visualization_info.json` 保存阈值 0.1、top-20 与前三帧的显示统计。原始日志中本机路径仅用于记录运行来源，复现时按上面的数据目录重新生成索引。

评估 JSON 中部分不适用的类别误差按 nuScenes 原始格式保留为 NaN；这不表示训练 Loss 出现非有限值。已上传日志的全部 Loss 记录均为有限值。
