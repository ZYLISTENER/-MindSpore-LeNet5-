# -MindSpore-LeNet5-
华东理工大学贯通式课题 案例2 基于MindSpore的手写字识别
23010022 周烨 人智233
本项目基于 MindSpore 框架实现 LeNet5 卷积神经网络，完成 MNIST 手写数字识别任务。
项目包含 Train.py（模型训练）和 Use.py（图片预处理 + 可视化交互界面）两个核心文件，支持上传彩色 / 花体手写数字图片，实时输出识别结果、置信度及概率分布可视化，通过自适应二值化、轮廓填充等预处理优化，解决了空心/艺术体数字识别不准的问题。

环境依赖：
Python ≥ 3.11
MindSpore ≥ 2.0.0
opencv-python ≥ 4.5.0
numpy ≥ 2.1.0
pillow ≥ 8.0.0
matplotlib ≥ 3.3.0
