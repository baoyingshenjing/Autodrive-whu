# 上游来源与本机适配

来源：https://github.com/Zhenye-Na/e2e-learning-self-driving-cars

基准提交：`ef999ee49acc443e284dbf2c38afc8ae47fd31c8`。MIT 许可见 [LICENSE](LICENSE)。本目录包含该提交的源码和课程实验中的本机适配版本。

本机修改位于 `src/RcCarDataset.py`、`src/drive.py`、`src/main.py` 和 `src/utils.py`，涉及图像读取、Windows 路径及推理服务兼容性。课程实际训练入口为上一级 `train.py`，沿用上游 `NetworkNvidia` 模型，替换了训练循环，并非上游原训练器的逐位复现。
