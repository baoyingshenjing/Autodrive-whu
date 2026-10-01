"""PETR 单卡教学配置（适配本机 8 GB 显存）。

保留官方 PETR 模型结构，缩小图片、查询数量和解码器层数来降低开销。
数据转换与评估统一使用 MMDetection3D 1.0.0rc6 的三维框约定；
不要直接加载旧版 PETR 的完整检测权重。当前 ResNet 骨干预训练权重可用。
通常只需在根目录运行 train.py --epochs 轮数，无需改动此文件。
"""
_base_ = ['../petr/petr_r50dcn_gridmask_p4.py']

# 三个入口会切换到 PETR-official，因此数据目录位于上一级。
data_root = '../v1.0-mini/'
class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
input_modality = dict(use_lidar=False, use_camera=True, use_radar=False,
                      use_map=False, use_external=False)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1., 1., 1.], to_rgb=False)
# 六路图片统一缩放裁剪到高 192、宽 512；教学版关闭随机翻转。
ida_aug_conf = dict(resize_lim=(0.32, 0.36), final_dim=(192, 512),
                    bot_pct_lim=(0.0, 0.0), rot_lim=(0.0, 0.0), H=900, W=1600,
                    rand_flip=False)
# 查询框数 100、深度采样数 32、解码器 3 层，减少显存和训练时间。
model = dict(
    img_backbone=dict(pretrained='ckpts/resnet50_msra-5891d200.pth'),
    pts_bbox_head=dict(num_query=100, depth_num=32,
        transformer=dict(decoder=dict(num_layers=3)),
        bbox_coder=dict(max_num=100)))

# 训练流水线：图片/标注加载 → 范围和类别过滤 → 图像增强 → 归一化 → 打包。
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    # 不使用旧版世界坐标旋转增强，避免与新版三维框约定混用。
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])]
# 测试流水线不加载训练标签；真实标签由 nuScenes 官方评估器读取。
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D', img_scale=(512, 192), pts_scale_ratio=1, flip=False,
         transforms=[dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
                     dict(type='Collect3D', keys=['img'])])]

# 每次处理一帧的六张图；Windows 下使用主进程读数据，避免多进程兼容问题。
data = dict(samples_per_gpu=1, workers_per_gpu=0,
    train=dict(_delete_=True, type='CustomNuScenesDataset', data_root=data_root,
               ann_file=data_root+'nuscenes_mini_infos_train.pkl', pipeline=train_pipeline,
               classes=class_names, modality=input_modality, test_mode=False,
               use_valid_flag=True, box_type_3d='LiDAR'),
    val=dict(_delete_=True, type='CustomNuScenesDataset', data_root=data_root,
             ann_file=data_root+'nuscenes_mini_infos_val.pkl', pipeline=test_pipeline,
             classes=class_names, modality=input_modality, test_mode=True, box_type_3d='LiDAR'),
    test=dict(_delete_=True, type='CustomNuScenesDataset', data_root=data_root,
              ann_file=data_root+'nuscenes_mini_infos_val.pkl', pipeline=test_pipeline,
              classes=class_names, modality=input_modality, test_mode=True, box_type_3d='LiDAR'))
# FP32 训练，学习率 0.0002；梯度裁剪有助于训练稳定。
optimizer = dict(lr=2e-4)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=20)
# 根目录 train.py 会用 --epochs 同步覆盖这两个轮数。
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)
# 每轮保存一次，最多保留三份；关闭 Windows 需要额外权限的符号链接。
checkpoint_config = dict(interval=1, create_symlink=False, max_keep_ckpts=3)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1, pipeline=test_pipeline)
work_dir = '../outputs/default'
gpu_ids = [0]
dist_params = dict(backend='gloo')
opencv_num_threads = 0
mp_start_method = 'spawn'
