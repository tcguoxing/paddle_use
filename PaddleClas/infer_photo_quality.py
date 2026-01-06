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
    def __init__(self, logs_dir, model_path, timestamp):
        """初始化分类日志记录器
        Args:
            logs_dir (str): 日志文件保存目录
            model_path (str): 模型文件路径
            timestamp (str): 推理时间戳
        """
        self.logs_dir = logs_dir
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.timestamp = timestamp
        
        # 生成日志文件名，包含时间戳
        log_file_name = f"classification_results_{timestamp}"
        
        # CSV日志文件路径
        self.csv_log_path = os.path.join(logs_dir, f"{log_file_name}.csv")
        
        # JSON日志文件路径
        self.json_log_path = os.path.join(logs_dir, f"{log_file_name}.json")
        
        # 初始化结果列表
        self.image_results = []
        self.processing_times = []
        
        # 写入CSV文件头，包含处理时间字段
        try:
            with open(self.csv_log_path, 'w') as f:
                f.write("timestamp,image_path,model_name,pred_class,confidence,good_confidence,bad_confidence,processing_time\n")
            info(f"CSV日志文件已创建: {self.csv_log_path}")
        except Exception as e:
            error(f"创建CSV日志文件失败: {e}")
            self.csv_log_path = None
    
    def log_result(self, image_path, result, processing_time=0.0):
        """记录单张图像的分类结果
        Args:
            image_path (str): 图像文件路径
            result (dict): 分类结果
            processing_time (float): 图像处理时间（秒）
        """
        # 获取当前时间戳
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 提取结果
        pred_class = result['pred_class']
        confidence = result['confidence']
        good_confidence = result['confidences']['good']
        bad_confidence = result['confidences']['bad']
        
        # 记录处理时间
        self.processing_times.append(processing_time)
        
        # 写入CSV文件
        if self.csv_log_path:
            try:
                with open(self.csv_log_path, 'a') as f:
                    f.write(f"{timestamp},{image_path},{self.model_name},{pred_class},{confidence},{good_confidence},{bad_confidence},{processing_time:.4f}\n")
            except Exception as e:
                error(f"写入CSV日志失败: {e}")
        
        # 添加到结果列表
        self.image_results.append({
            'timestamp': timestamp,
            'image_path': image_path,
            'model_name': self.model_name,
            'pred_class': pred_class,
            'confidence': confidence,
            'confidences': result['confidences'],
            'processing_time': processing_time
        })
    
    def save_json(self, start_time, end_time, total_images, good_count, bad_count):
        """保存JSON格式日志
        Args:
            start_time (datetime): 推理开始时间
            end_time (datetime): 推理结束时间
            total_images (int): 总处理图像数量
            good_count (int): 好照片数量
            bad_count (int): 坏照片数量
        """
        # 计算平均处理时间
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
        
        # 构建结构化JSON日志
        json_log = {
            "inference_info": {
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "timestamp": self.timestamp
            },
            "model_info": {
                "model_path": self.model_path,
                "model_name": self.model_name
            },
            "statistics": {
                "total_images": total_images,
                "good_images": good_count,
                "bad_images": bad_count,
                "good_ratio": good_count / total_images if total_images > 0 else 0.0,
                "bad_ratio": bad_count / total_images if total_images > 0 else 0.0,
                "avg_processing_time_seconds": avg_processing_time,
                "total_processing_time_seconds": sum(self.processing_times)
            },
            "image_results": self.image_results
        }
        
        # 保存JSON日志
        try:
            import json
            with open(self.json_log_path, 'w') as f:
                json.dump(json_log, f, indent=2, ensure_ascii=False)
            info(f"JSON日志已保存到: {self.json_log_path}")
        except Exception as e:
            error(f"保存JSON日志失败: {e}")

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
    # 记录推理开始时间
    start_time = datetime.datetime.now()
    
    # 1. 生成主文件夹名称和路径
    # 获取当前时间戳，格式：YYYYMMDD_HHMMSS
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    # 从模型路径中提取模型名称
    model_name = os.path.basename(args.model_path)
    # 移除模型文件扩展名
    model_name_without_ext = os.path.splitext(model_name)[0]
    # 构建主文件夹名称：评估模型名称+时间戳
    main_folder_name = f"{model_name_without_ext}_{timestamp}"
    # 构建主输出目录，支持绝对路径和相对路径
    main_output_dir = os.path.abspath(os.path.join(args.output_dir, main_folder_name))
    
    # 2. 创建标准化目录结构
    try:
        # 创建主文件夹
        os.makedirs(main_output_dir, exist_ok=True)
        
        # 创建三个必要子文件夹：images、logs和configs
        images_dir = os.path.join(main_output_dir, "images")
        logs_dir = os.path.join(main_output_dir, "logs")
        configs_dir = os.path.join(main_output_dir, "configs")
        
        # 创建子文件夹，确保原子性
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)
        
        # 在images文件夹下创建good和bad二级子文件夹
        good_images_dir = os.path.join(images_dir, "good")
        bad_images_dir = os.path.join(images_dir, "bad")
        os.makedirs(good_images_dir, exist_ok=True)
        os.makedirs(bad_images_dir, exist_ok=True)
        
        info(f"输出目录结构已创建: {main_output_dir}")
        info(f"好照片将保存到: {good_images_dir}")
        info(f"坏照片将保存到: {bad_images_dir}")
        info(f"日志文件将保存到: {logs_dir}")
        info(f"配置参数将保存到: {configs_dir}")
    except Exception as e:
        error(f"创建目录结构失败: {e}")
        error(f"请检查权限、路径或磁盘空间")
        return []
    
    # 3. 加载模型
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
    logger = ClassificationLogger(logs_dir, args.model_path, timestamp)
    
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
                # 记录单张图像的开始处理时间
                img_start_time = datetime.datetime.now()
                
                result = future.result()
                
                # 计算单张图像处理时间
                img_end_time = datetime.datetime.now()
                processing_time = (img_end_time - img_start_time).total_seconds()
                
                if result:
                    results.append(result)
                    
                    # 打印结果到控制台
                    info(
                        f"图像: {os.path.basename(img_path)} -> "
                        f"分类: {result['pred_class']}，" 
                        f"置信度: {result['confidence']:.4f}"
                    )
                    
                    # 记录结果到日志文件，包含处理时间
                    logger.log_result(img_path, result, processing_time)
                    
                    # 保存照片到对应的分类文件夹，默认保存，除非指定--no_save_good
                    save_path = None
                    if not args.no_save_good:
                        if result['pred_class'] == 'good':
                            save_path = save_good_image(result, good_images_dir)
                        elif result['pred_class'] == 'bad':
                            save_path = save_good_image(result, bad_images_dir)
                        
                        if save_path:
                            info(f"已保存照片到: {save_path}")
            except Exception as e:
                error(f"处理图像 {img_path} 时发生异常: {e}")
            finally:
                progress_bar.update(1)
        progress_bar.close()
    
    # 记录推理结束时间
    end_time = datetime.datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # 7. 统计结果
    good_count = sum(1 for r in results if r['pred_class'] == 'good')
    bad_count = sum(1 for r in results if r['pred_class'] == 'bad')
    total_images = len(results)
    avg_processing_time = total_duration / total_images if total_images > 0 else 0.0
    
    if results:
        info("\n" + "="*50)
        info(f"处理完成，共 {total_images} 张图像")
        info(f"好照片 (good): {good_count} 张")
        info(f"坏照片 (bad): {bad_count} 张")
        info(f"好照片占比: {good_count / total_images * 100:.2f}%")
        info(f"总处理时间: {total_duration:.2f} 秒")
        info(f"平均处理时间: {avg_processing_time:.4f} 秒/张")
        info("="*50)
    
    # 8. 保存JSON日志文件，包含完整的统计信息
    logger.save_json(start_time, end_time, total_images, good_count, bad_count)
    
    # 9. 保存配置参数到configs文件夹
    configs = {
        "model_path": args.model_path,
        "input_path": args.input_path,
        "output_dir": args.output_dir,
        "save_good": not args.no_save_good,
        "device": device,
        "model_name": model_name,
        "timestamp": timestamp,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "paddle_version": paddle.__version__
    }
    
    config_file = os.path.join(configs_dir, "config.json")
    try:
        import json
        with open(config_file, 'w') as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
        info(f"配置参数已保存到: {config_file}")
    except Exception as e:
        error(f"保存配置参数失败: {e}")
    
    # 10. 生成标准化摘要文件summary.txt
    summary_file = os.path.join(main_output_dir, "summary.txt")
    try:
        with open(summary_file, 'w') as f:
            f.write("# 照片质量评估推理摘要\n\n")
            f.write("## 1. 推理基本信息\n")
            f.write(f"- 推理开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 推理结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 总处理时间: {total_duration:.2f} 秒\n")
            f.write(f"- 平均处理时间: {avg_processing_time:.4f} 秒/张\n\n")
            
            f.write("## 2. 模型信息\n")
            f.write(f"- 模型路径: {args.model_path}\n")
            f.write(f"- 模型名称: {model_name}\n")
            f.write(f"- Paddle版本: {paddle.__version__}\n\n")
            
            f.write("## 3. 推理参数\n")
            f.write(f"- 输入路径: {args.input_path}\n")
            f.write(f"- 输出路径: {main_output_dir}\n")
            f.write(f"- 设备: {device}\n\n")
            
            f.write("## 4. 统计结果\n")
            f.write(f"- 总处理照片数量: {total_images}\n")
            f.write(f"- 判定为好照片的数量: {good_count}\n")
            f.write(f"- 判定为坏照片的数量: {bad_count}\n")
            f.write(f"- 好照片占比: {good_count / total_images * 100:.2f}%\n")
            f.write(f"- 坏照片占比: {bad_count / total_images * 100:.2f}%\n\n")
            
            f.write("## 5. 输出目录结构\n")
            f.write(f"- 主文件夹: {main_output_dir}\n")
            f.write(f"- 好照片存储: {os.path.join(main_output_dir, 'images', 'good')}\n")
            f.write(f"- 坏照片存储: {os.path.join(main_output_dir, 'images', 'bad')}\n")
            f.write(f"- 日志文件: {logs_dir}\n")
            f.write(f"- 配置参数: {configs_dir}\n")
        
        info(f"推理摘要已保存到: {summary_file}")
    except Exception as e:
        error(f"生成摘要文件失败: {e}")
    
    return results

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='照片质量分类推理脚本')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重文件路径')
    parser.add_argument('--input_path', type=str, required=True, help='输入图像路径，可以是单个文件或目录')
    parser.add_argument('--output_dir', type=str, default='./good_photos', help='好照片保存目录')
    parser.add_argument('--no_save_good', action='store_true', help='不保存照片到分类目录，默认保存')
    
    args = parser.parse_args()
    
    # 执行推理
    infer(args)

if __name__ == '__main__':
    main()
