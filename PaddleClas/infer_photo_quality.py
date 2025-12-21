#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于MobileNetV3_large的照片质量分类推理脚本
用于区分清晰对焦（good）和失焦模糊（bad）照片
针对Apple M3 Pro处理器优化
"""

import os
import argparse
import logging
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import paddle
from paddle.vision import transforms as T

# 直接从ppcls包导入，避免通过paddleclas.py的相对导入
import sys
import os
# 将当前目录添加到sys.path，确保能找到ppcls包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppcls.arch import backbone

# 初始化PaddleClas的logger
from ppcls.utils.logger import init_logger, info, warning, error, debug

# 导入时间模块，用于生成时间戳
import datetime

# 自定义日志文件记录类
class ClassificationLogger:
    def __init__(self, output_dir, model_path):
        self.output_dir = output_dir
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成带时间戳的日志文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV日志文件路径
        self.csv_log_path = os.path.join(output_dir, f"{self.model_name}_classification_results_{timestamp}.csv")
        
        # JSON日志文件路径
        self.json_log_path = os.path.join(output_dir, f"{self.model_name}_classification_results_{timestamp}.json")
        
        # 初始化JSON结果列表
        self.json_results = []
        
        # 写入CSV文件头
        with open(self.csv_log_path, 'w') as f:
            f.write("timestamp,image_path,model_name,pred_class,confidence,good_confidence,bad_confidence\n")
        
        info(f"日志文件已创建: {self.csv_log_path}")
        info(f"日志文件已创建: {self.json_log_path}")
    
    def log_result(self, image_path, result):
        # 获取当前时间戳
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 提取结果
        pred_class = result['pred_class']
        confidence = result['confidence']
        good_confidence = result['confidences']['good']
        bad_confidence = result['confidences']['bad']
        
        # 写入CSV文件
        with open(self.csv_log_path, 'a') as f:
            f.write(f"{timestamp},{image_path},{self.model_name},{pred_class},{confidence},{good_confidence},{bad_confidence}\n")
        
        # 添加到JSON结果列表
        json_entry = {
            'timestamp': timestamp,
            'image_path': image_path,
            'model_name': self.model_name,
            'pred_class': pred_class,
            'confidence': confidence,
            'confidences': result['confidences']
        }
        self.json_results.append(json_entry)
    
    def save_json(self):
        # 保存JSON结果
        import json
        with open(self.json_log_path, 'w') as f:
            json.dump(self.json_results, f, indent=2, ensure_ascii=False)
        info(f"JSON日志已保存到: {self.json_log_path}")

# 初始化PaddleClas的logger
init_logger()

# 检查设备并设置为GPU优先
paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
device = paddle.get_device()
info(f"当前设备: {device}")

# 针对Apple M3 Pro优化设置
# 启用内存优化
paddle.set_flags({'FLAGS_allocator_strategy': 'auto_growth'})

# 分类标签
CLASS_NAMES = ['good', 'bad']

def get_transforms():
    """获取图像变换
    与训练阶段保持一致的预处理流程
    Returns:
        transform: 图像变换组合
    """
    return T.Compose([
        # 将图像调整为512*512
        T.Resize((512, 512)),
        # 转换为张量
        T.ToTensor(),
        # 归一化，使用ImageNet均值和标准差
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_model(num_classes=2, pretrained=False):
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

def load_model(model_path):
    """加载训练好的模型
    Args:
        model_path (str): 模型权重文件路径
    Returns:
        model: 加载好权重的模型
    """
    info(f"正在加载模型: {model_path}")
    
    # 创建模型
    model = create_model(num_classes=2, pretrained=False)
    
    # 加载权重
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)
    
    # 设置为评估模式
    model.eval()
    
    info("模型加载完成")
    return model

def process_single_image(image_path, model, transform):
    """处理单张图像
    Args:
        image_path (str): 图像文件路径
        model (paddle.nn.Layer): 加载好的模型
        transform (callable): 图像变换函数
    Returns:
        dict: 分类结果，包含图像路径、分类标签、置信度
    """
    try:
        # 读取图像
        # 使用cv2读取图像，替代paddle.vision.image.load_image
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"无法读取图像文件: {image_path}")
        # 转换颜色空间 BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
        
        # 模型推理
        with paddle.no_grad():
            output = model(image_tensor)
            prob = paddle.nn.functional.softmax(output, axis=1)
            confidence = prob.numpy()[0]
            pred_label = paddle.argmax(prob, axis=1).numpy()[0]
        
        # 构造结果
        result = {
            'image_path': image_path,
            'pred_label': int(pred_label),
            'pred_class': CLASS_NAMES[int(pred_label)],
            'confidence': float(confidence[int(pred_label)]),
            'confidences': {
                'good': float(confidence[0]),
                'bad': float(confidence[1])
            }
        }
        
        return result
    except Exception as e:
        error(f"处理图像 {image_path} 失败: {e}")
        return None

def save_good_image(result, output_dir):
    """保存好照片到指定目录
    Args:
        result (dict): 图像分类结果
        output_dir (str): 输出目录
    """
    if result['pred_class'] == 'good':
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 构造输出路径
        filename = os.path.basename(result['image_path'])
        output_path = os.path.join(output_dir, filename)
        
        # 复制文件
        shutil.copy2(result['image_path'], output_path)
        return output_path
    return None

def infer(args):
    """推理主函数
    Args:
        args: 命令行参数
    """
    # 1. 加载模型
    model = load_model(args.model_path)
    
    # 2. 获取图像变换
    transform = get_transforms()
    
    # 3. 获取图像列表
    image_paths = []
    
    # 处理单个文件
    if os.path.isfile(args.input_path):
        if args.input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths = [args.input_path]
        else:
            error(f"不支持的文件格式: {args.input_path}")
            return
    # 处理目录
    elif os.path.isdir(args.input_path):
        info(f"正在扫描目录: {args.input_path}")
        for filename in os.listdir(args.input_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(args.input_path, filename))
        info(f"共找到 {len(image_paths)} 张图像")
    else:
        error(f"输入路径无效: {args.input_path}")
        return
    
    if not image_paths:
        error("没有找到可处理的图像文件")
        return
    
    # 4. 初始化分类日志记录器
    logger = ClassificationLogger(args.output_dir, args.model_path)
    
    # 5. 批量处理图像
    results = []
    
    # 针对6M-10M高分辨率照片的内存优化：使用生成器加载
    def image_generator():
        for img_path in image_paths:
            yield img_path
    
    # 使用线程池加速处理，针对Apple M3 Pro优化
    # 增加max_workers数量，充分利用M3 Pro的多核性能
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有任务
        future_to_image = {
            executor.submit(process_single_image, img_path, model, transform): img_path 
            for img_path in image_generator()
        }
        
        # 处理结果
        progress_bar = tqdm(total=len(image_paths), desc="推理")
        for future in as_completed(future_to_image):
            img_path = future_to_image[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    
                    # 打印结果到控制台
                    info(
                        f"图像: {os.path.basename(img_path)} -> "
                        f"分类: {result['pred_class']}，" 
                        f"置信度: {result['confidence']:.4f}"
                    )
                    
                    # 记录结果到日志文件
                    logger.log_result(img_path, result)
                    
                    # 保存好照片
                    if args.save_good:
                        save_path = save_good_image(result, args.output_dir)
                        if save_path:
                            info(f"已保存好照片到: {save_path}")
            except Exception as e:
                error(f"处理图像 {img_path} 时发生异常: {e}")
            finally:
                progress_bar.update(1)
        progress_bar.close()
    
    # 6. 保存JSON日志文件
    logger.save_json()
    
    # 7. 统计结果
    if results:
        good_count = sum(1 for r in results if r['pred_class'] == 'good')
        bad_count = sum(1 for r in results if r['pred_class'] == 'bad')
        
        info("\n" + "="*50)
        info(f"处理完成，共 {len(results)} 张图像")
        info(f"好照片 (good): {good_count} 张")
        info(f"坏照片 (bad): {bad_count} 张")
        info(f"好照片占比: {good_count / len(results) * 100:.2f}%")
        info("="*50)
    
    return results

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='照片质量分类推理脚本')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重文件路径')
    parser.add_argument('--input_path', type=str, required=True, help='输入图像路径，可以是单个文件或目录')
    parser.add_argument('--output_dir', type=str, default='./good_photos', help='好照片保存目录')
    parser.add_argument('--save_good', action='store_true', help='是否保存好照片到指定目录')
    
    args = parser.parse_args()
    
    # 执行推理
    infer(args)

if __name__ == '__main__':
    main()
