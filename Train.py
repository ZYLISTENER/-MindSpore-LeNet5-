# -*- coding: utf-8 -*-
"""
MindSpore MNIST训练脚本（适配推理脚本版）
=========================================
功能描述：
    基于MindSpore框架训练LeNet5卷积神经网络，用于MNIST手写数字分类任务。
    脚本严格对齐推理脚本的网络结构、输入尺寸和数据预处理逻辑，确保训练出的模型可直接用于推理。

核心特点：
    1. 网络结构：与推理脚本完全一致的LeNet5实现
    2. 输入尺寸：统一使用32x32（MNIST原始28x28需缩放）
    3. 预处理：与推理脚本100%对齐的归一化/维度调整逻辑
    4. 兼容性：CPU训练，无需GPU/昇腾环境

环境依赖：
    - mindspore >= 2.0.0
    - numpy >= 1.21.0
    - 已下载MNIST数据集（需配置数据集路径）

使用方法：
    1. 配置DATASET_ROOT为MNIST数据集根目录
    2. 调整训练超参数（可选）
    3. 直接运行脚本，模型会保存到./checkpoints目录
"""

import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, set_device, load_checkpoint, load_param_into_net
from mindspore.train import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.dataset import MnistDataset
from mindspore.dataset import transforms
from mindspore.dataset.vision import Rescale, Resize, HWC2CHW

# ======================== 1. 环境配置（全局参数） ========================
"""
环境配置说明：
- mode: PYNATIVE_MODE（动态图）适合调试，GRAPH_MODE（静态图）训练速度更快
- device_target: CPU/GPU/Ascend，此处选择CPU保证兼容性
"""
# 设置MindSpore运行模式和设备
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
set_device("CPU")  # 绑定CPU设备

# ---------------------- 路径配置 ----------------------
# MNIST数据集根目录（请替换为你的数据集实际路径）
# 数据集结构：DATASET_ROOT/train（训练集）、DATASET_ROOT/test（测试集）
DATASET_ROOT = r"D:\mindspore\mnist"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")  # 训练集路径
TEST_DIR = os.path.join(DATASET_ROOT, "test")    # 测试集路径

# ---------------------- 训练超参数 ----------------------
"""
训练超参数说明：
- EPOCHS: 训练轮数，5轮可满足MNIST基本精度需求
- BATCH_SIZE: 批次大小，64是CPU训练的合适值（避免内存溢出）
- LEARNING_RATE: 学习率，0.01配合Momentum优化器效果较好
- TARGET_SIZE: 输入图片尺寸，必须与推理脚本保持一致（32x32）
"""
EPOCHS = 5                # 训练轮数
BATCH_SIZE = 64           # 批次大小
LEARNING_RATE = 0.01      # 学习率
TARGET_SIZE = 32          # 输入图片目标尺寸（与推理脚本严格对齐）

# ======================== 2. 数据加载与预处理 ========================
def create_dataset(dataset_path: str, batch_size: int = 32, is_train: bool = True) -> ms.dataset.Dataset:
    """
    创建MNIST数据集加载器，预处理逻辑与推理脚本100%对齐
    
    Args:
        dataset_path: 数据集目录路径（train/test）
        batch_size: 批次大小
        is_train: 是否为训练集（训练集开启shuffle，测试集关闭）
    
    Returns:
        预处理后的MindSpore Dataset对象
    
    预处理流程（与推理脚本严格匹配）：
        1. Resize: 28x28（MNIST原始）→ 32x32（目标尺寸）
        2. Rescale: 像素值归一化到0-1区间（仅除以255，无均值/标准差）
        3. HWC2CHW: 维度转换 [H,W,C] → [C,H,W]（适配MindSpore输入格式）
    """
    # 加载MNIST数据集（usage="all"表示加载所有数据）
    dataset = MnistDataset(dataset_dir=dataset_path, usage="all", shuffle=False)
    
    # 定义图片预处理流水线（核心：与推理脚本对齐）
    img_transforms = [
        Resize((TARGET_SIZE, TARGET_SIZE)),  # 缩放至32x32（推理脚本输入尺寸）
        Rescale(1.0 / 255.0, 0),            # 归一化：像素值/255，无偏移量
        HWC2CHW()                           # 维度调整：高度→宽度→通道 → 通道→高度→宽度
    ]
    
    # 标签类型转换（转为int32）
    label_transform = transforms.TypeCast(ms.int32)
    
    # 应用预处理到数据集
    dataset = dataset.map(operations=img_transforms, input_columns="image")  # 图片预处理
    dataset = dataset.map(operations=label_transform, input_columns="label") # 标签预处理
    
    # 训练集开启随机打乱（提升泛化性），测试集无需打乱
    if is_train:
        dataset = dataset.shuffle(buffer_size=10000)  # 打乱缓冲区大小
    
    # 分批次，drop_remainder=True确保批次大小一致
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset

# 加载训练集和测试集
train_dataset = create_dataset(TRAIN_DIR, BATCH_SIZE, is_train=True)
test_dataset = create_dataset(TEST_DIR, BATCH_SIZE, is_train=False)

# ======================== 3. LeNet5模型定义（与推理脚本完全一致） ========================
class LeNet5(nn.Cell):
    """
    LeNet5卷积神经网络实现（与推理脚本严格一致）
    
    网络结构（适配32x32输入）：
        Input(1x32x32) → Conv1(6x28x28) → ReLU → MaxPool(6x14x14)
        → Conv2(16x10x10) → ReLU → MaxPool(16x5x5) → Flatten(400)
        → FC1(120) → ReLU → FC2(84) → ReLU → FC3(10) → Output
    
    关键注意点：
        - 卷积层pad_mode='valid'（无填充），与推理脚本一致
        - 全连接层维度16*5*5=400，需严格匹配卷积层输出
    """
    def __init__(self, num_class: int = 10, num_channel: int = 1):
        """
        初始化LeNet5网络层
        
        Args:
            num_class: 分类类别数，默认10（0-9数字）
            num_channel: 输入通道数，默认1（灰度图）
        """
        super(LeNet5, self).__init__()
        
        # 卷积层：特征提取
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')  # 1→6通道，5x5卷积核，无填充
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')           # 6→16通道，5x5卷积核，无填充
        
        # 全连接层：特征映射到分类结果
        self.fc1 = nn.Dense(16 * 5 * 5, 120)  # 16*5*5=400（卷积层输出特征数）→120
        self.fc2 = nn.Dense(120, 84)          # 120→84
        self.fc3 = nn.Dense(84, num_class)    # 84→10（最终分类）
        
        # 激活函数与池化层
        self.relu = nn.ReLU()                          # ReLU非线性激活
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        self.flatten = nn.Flatten()                    # 多维特征展平为一维

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        网络前向传播逻辑（MindSpore特有，替代forward方法）
        
        Args:
            x: 输入张量，形状[batch_size, channel, height, width]
        
        Returns:
            输出张量，形状[batch_size, num_class]（各类别预测得分）
        """
        # 第一层卷积→激活→池化
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        
        # 第二层卷积→激活→池化
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        
        # 展平→全连接层→输出
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x

# 初始化LeNet5模型
network = LeNet5()

# ======================== 4. 训练配置 ========================
"""
训练配置说明：
- 损失函数：SoftmaxCrossEntropyWithLogits（适配分类任务，sparse=True表示标签为稀疏格式）
- 优化器：Momentum（带动量的梯度下降，收敛更快）
- 评估指标：accuracy（分类准确率）
"""
# 损失函数：Softmax交叉熵（适配分类任务）
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# 优化器：Momentum（配合0.01学习率效果最优）
optimizer = nn.Momentum(network.trainable_params(), learning_rate=LEARNING_RATE, momentum=0.9)

# 评估指标：准确率
metrics = {"accuracy"}

# 构建训练模型
model = Model(network, loss_fn, optimizer, metrics=metrics)

# ---------------------- 模型保存配置 ----------------------
"""
模型保存配置：
- save_checkpoint_steps: 每937步保存一次（对应1个epoch：60000样本/64批次≈937步）
- keep_checkpoint_max: 最多保留5个检查点文件（避免磁盘占用过大）
- prefix: 模型文件名前缀
- directory: 模型保存目录（自动创建）
"""
# 创建checkpoints目录（不存在则创建）
os.makedirs("./checkpoints", exist_ok=True)

# 检查点配置
ckpt_config = CheckpointConfig(
    save_checkpoint_steps=937,  # 每937步保存一次（1个epoch）
    keep_checkpoint_max=5       # 最多保留5个模型文件
)

# 模型保存回调
ckpt_callback = ModelCheckpoint(
    prefix="lenet_mnist_final",  # 模型文件名前缀
    directory="./checkpoints",   # 保存目录
    config=ckpt_config
)

# 训练回调函数列表
"""
回调函数说明：
- TimeMonitor: 监控每批次训练时间
- LossMonitor(50): 每50步打印一次损失值
- ckpt_callback: 模型保存回调
"""
callbacks = [TimeMonitor(), LossMonitor(50), ckpt_callback]

# ======================== 5. 训练与评估（主执行逻辑） ========================
if __name__ == "__main__":
    """
    主执行流程：
    1. 打印训练配置信息
    2. 执行模型训练（dataset_sink_mode=False适配CPU）
    3. 测试集评估，输出准确率
    """
    # 打印训练信息
    print("="*60)
    print(f"训练开始 | 数据集路径: {DATASET_ROOT} | 输入尺寸: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"超参数 | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Learning Rate: {LEARNING_RATE}")
    print("="*60)
    
    # 开始训练（dataset_sink_mode=False：CPU训练必须关闭数据下沉）
    model.train(EPOCHS, train_dataset, callbacks=callbacks, dataset_sink_mode=False)
    
    # 测试集评估
    print("\n开始评估测试集...")
    eval_result = model.eval(test_dataset, dataset_sink_mode=False)
    
    # 打印评估结果
    print(f"\n测试集准确率: {eval_result['accuracy']:.4f}")
    print("\n训练完成！模型保存在 ./checkpoints 目录下")
    print("="*60)