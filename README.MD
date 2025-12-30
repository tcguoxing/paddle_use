本项目是一个基于PaddleClas的照片质量分类系统，主要用于区分清晰对焦（good）和失焦模糊（bad）的照片。

## 项目结构与功能
### 核心脚本
1. 训练脚本 ( train_photo_quality.py )
   
   - 使用MobileNetV3_large_x1_0模型进行训练
   - 支持通过命令行参数配置模型的预训练状态
   - 实现了数据平衡采样和混合精度训练
   - 支持多进程数据加载和Apple M3 Pro处理器优化
2. 推理脚本 ( infer_photo_quality.py )
   
   - 加载训练好的模型对照片进行质量分类
   - 将分类结果保存为CSV和JSON格式的日志文件
   - 自动将分类为good的照片保存到指定目录
   - 生成推理摘要信息和配置参数文件
### 技术栈
- 深度学习框架 ：PaddlePaddle/PaddleClas
- 模型架构 ：MobileNetV3_large_x1_0
- 图像处理 ：OpenCV
- 命令行解析 ：argparse
- 日志记录 ：Python logging模块
- 并发处理 ：ThreadPoolExecutor
### 工作流程
1. 数据准备 ：
   
   - 照片按类别存放在 good 和 bad 两个子目录中
   - 支持JPG、JPEG、PNG格式的图像文件
2. 模型训练 ：
   
   - 加载数据集并进行图像预处理（调整大小、翻转、颜色变换等）
   - 使用平衡采样器处理类别不平衡问题
   - 支持冻结预训练模型的特征提取层，只训练分类头
   - 保存最佳模型和定期保存模型
3. 照片质量推理 ：
   
   - 加载训练好的模型权重
   - 对输入图像进行分类预测
   - 将分类结果保存到日志文件
   - 将分类为good的照片保存到指定目录
   - 生成推理摘要和配置参数文件
### 输出结果组织
- 推理结果保存在 good_photos 目录下以模型名称命名的文件夹中
- 文件夹内包含：
  - images/ ：保存分类为good的照片
  - logs/ ：保存CSV和JSON格式的日志文件
  - configs/ ：保存配置参数
  - summary.txt ：保存推理摘要信息
### 命令行使用
- 训练 ： python train_photo_quality.py --data_dir <数据集目录> --output_dir <输出目录> --pretrained <是否使用预训练权重>
- 推理 ： python infer_photo_quality.py --model_path <模型路径> --input_path <输入图像或目录> --output_dir <输出目录>
本项目旨在提供一个高效、易用的照片质量分类解决方案，可用于照片筛选、质量控制等场景。