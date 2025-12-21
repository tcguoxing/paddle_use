#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于MobileNetV3_large的照片质量分类训练脚本
用于区分清晰对焦（good）和失焦模糊（bad）照片
针对Apple M3 Pro处理器优化
"""

import os
import argparse
import logging
import time
from tqdm import tqdm

import paddle
from paddle.io import Dataset, DataLoader, random_split
from paddle.vision import transforms as T
from paddle.nn import functional as F

# 直接从ppcls包导入，避免通过paddleclas.py的相对导入
import sys
import os
# 将当前目录添加到sys.path，确保能找到ppcls包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppcls.arch import backbone
from ppcls.utils import logger

# 初始化PaddleClas的logger
from ppcls.utils.logger import init_logger, info, warning, error, debug
init_logger()

# 使用PaddleClas的logger
info('开始初始化训练脚本')

# 检查设备并设置为GPU优先
paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
device = paddle.get_device()
info(f"当前设备: {device}")

# 针对Apple M3 Pro优化设置
# 启用内存优化
paddle.set_flags({'FLAGS_allocator_strategy': 'auto_growth'})

class PhotoQualityDataset(Dataset):
    """照片质量数据集类
    用于加载和处理good/bad二分类照片数据
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): 数据集根目录，包含good和bad两个子目录
            transform (callable, optional): 图像变换函数
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载数据路径和标签"""
        info(f'开始加载数据集 {self.data_dir}')
        # good类标签为0，bad类标签为1
        for label, class_name in enumerate(['good', 'bad']):
            class_dir = os.path.join(self.data_dir, class_name)
            info(f"开始加载类别 {class_name}，目录: {class_dir}")
            if not os.path.exists(class_dir):
                warning(f"目录 {class_dir} 不存在，跳过")
                continue
            
            # 获取所有图像文件
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_dir, filename)
                    self.image_paths.append(image_path)
                    self.labels.append(label)
        
        info(f"加载完成: 共 {len(self.image_paths)} 张图像，其中 good: {sum(1 for l in self.labels if l == 0)}, bad: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        try:
            # 读取图像
            image_path = self.image_paths[idx]
            # 使用cv2读取图像，替代paddle.vision.image.load_image
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"无法读取图像文件: {image_path}")
            # 转换颜色空间 BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            label = self.labels[idx]
            return image, label
        except Exception as e:
            error(f"读取图像 {self.image_paths[idx]} 失败: {e}")
            # 返回随机图像和标签，避免训练中断
            return paddle.rand([3, 512, 512]), self.labels[idx]

import numpy as np
import cv2
from datetime import datetime

class GaussianBlur:
    """自定义高斯模糊变换，用于替代paddle.vision.transforms.GaussianBlur
    Args:
        kernel_size (tuple): 高斯核大小，必须为奇数
        sigma (tuple): 高斯核标准差范围
    """
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, img):
        """应用高斯模糊
        Args:
            img (numpy.ndarray): 输入图像，HWC格式
        Returns:
            numpy.ndarray: 模糊后的图像
        """
        # 生成随机标准差
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        # 应用高斯模糊
        return cv2.GaussianBlur(img, self.kernel_size, sigma)

def get_transforms(is_train=True):
    """获取图像变换
    Args:
        is_train (bool): 是否为训练集
    Returns:
        transform: 图像变换组合
    """
    if is_train:
        return T.Compose([
            # 将图像调整为512*512
            T.Resize((512, 512)),
            # 随机水平翻转
            T.RandomHorizontalFlip(0.5),
            # 随机垂直翻转
            T.RandomVerticalFlip(0.2),
            # 随机亮度、对比度、饱和度调整
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # 使用自定义高斯模糊替代T.GaussianBlur
            GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            # 转换为张量
            T.ToTensor(),
            # 归一化，使用ImageNet均值和标准差
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            # 验证集只进行必要的变换
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_model(num_classes=2, pretrained=True):
    """创建MobileNetV3_large模型
    Args:
        num_classes (int): 分类数量
        pretrained (bool): 是否使用预训练权重
    Returns:
        model: 定义好的模型
    """
    # 创建MobileNetV3_large_x1_0模型
    model = backbone.MobileNetV3_large_x1_0(pretrained=pretrained)
    
    # 获取最后一层的输入特征数
    # MobileNetV3没有avg_pool_channel属性，使用class_expand属性代替
    in_features = model.class_expand
    
    # 替换分类头
    model.fc = paddle.nn.Linear(in_features, num_classes)
    
    return model

def train():
    """训练主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='照片质量分类训练脚本')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小，针对M3 Pro优化')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--use_amp', type=bool, default=True, help='是否使用混合精度训练，针对M3 Pro优化')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--pretrained', type=bool, default=True, help='是否使用预训练权重')
    args = parser.parse_args()
    
    # 设置随机种子
    paddle.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 数据加载
    info("正在加载数据集...")
    
    # 创建数据集
    full_dataset = PhotoQualityDataset(
        data_dir=args.data_dir,
        transform=get_transforms(is_train=True)
    )
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 设置验证集变换
    val_dataset.transform = get_transforms(is_train=False)
    
    # 创建数据加载器
    # 使用多进程加载，针对Apple M3 Pro优化
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # 针对M3 Pro优化的进程数
        use_shared_memory=True,
        drop_last=True,  # 丢弃最后一个不完整批次，提高训练稳定性
        timeout=120  # 增加超时时间，避免多进程加载超时
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        use_shared_memory=True,
        drop_last=False
    )
    
    # 2. 模型创建
    info("正在创建模型...")
    model = create_model(num_classes=2, pretrained=args.pretrained)
    
    # 3. 优化器配置
    # 针对Apple Silicon优化的AdamW配置
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.lr,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    )
    
    # 启用混合精度训练，提高训练速度
    # 针对Apple M3 Pro优化的混合精度配置
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024, use_dynamic_loss_scaling=True)
    use_amp = args.use_amp  # 使用命令行参数控制是否启用混合精度
    
    # 4. 损失函数
    criterion = paddle.nn.CrossEntropyLoss()
    
    # 5. 训练循环
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        info(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        train_bar = tqdm(train_loader, desc="训练")
        for batch_id, (images, labels) in enumerate(train_bar):
            # 使用混合精度训练
            with paddle.amp.auto_cast(enable=use_amp):
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # 反向传播，使用混合精度
            scaled = scaler.scale(loss)
            scaled.backward()
            
            # 优化器步骤
            scaler.minimize(optimizer, scaled)
            optimizer.clear_grad()
            
            # 计算指标
            # loss.numpy()返回0维数组，直接获取标量值
            train_loss += loss.numpy().item()
            acc = paddle.metric.accuracy(outputs, labels.unsqueeze(1))
            train_acc += acc.numpy().item()
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': f'{loss.numpy().item():.4f}',
                'acc': f'{acc.numpy().item():.4f}'
            })
        
        # 计算平均训练指标
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        val_bar = tqdm(val_loader, desc="验证")
        with paddle.no_grad():
            for images, labels in val_bar:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.numpy().item()
                acc = paddle.metric.accuracy(outputs, labels.unsqueeze(1))
                val_acc += acc.numpy().item()
                
    # 更新进度条
            val_bar.set_postfix({
                'loss': f'{loss.numpy().item():.4f}',
                'acc': f'{acc.numpy().item():.4f}'
            })
        
        # 计算平均验证指标
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        info(f"训练结果 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        info(f"验证结果 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            now = datetime.now()
            model_path = os.path.join(args.output_dir, f'best_model_epoch{epoch+1}_acc{val_acc:.4f}_{now.strftime("%Y%m%d_%H%M")}.pdparams')
            paddle.save(model.state_dict(), model_path)
            info(f"保存最佳模型到 {model_path}")
        
        # 每5轮保存一次模型
        if (epoch + 1) % 5 == 0:
            model_path = os.path.join(args.output_dir, f'model_epoch{epoch+1}_{now.strftime("%Y%m%d_%H%M")}.pdparams')
            paddle.save(model.state_dict(), model_path)
            info(f"保存模型到 {model_path}")
    
    info(f"训练完成，最佳验证准确率: {best_acc:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, f'final_model_{now.strftime("%Y%m%d_%H%M")}.pdparams')
    paddle.save(model.state_dict(), final_model_path)
    info(f"保存最终模型到 {final_model_path}")

if __name__ == '__main__':
    train()
