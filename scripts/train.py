#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import torch
import os
from datetime import datetime

def train_model(
    data='',
    epochs=100,
    batch_size=16,
    img_size=640,
    resume=False,
    device='',
    pretrained='yolov8n.pt',
    project='runs/train',
    name='exp'
):
    """
    使用YOLOv8训练模型
    
    参数:
        data: 数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像尺寸
        resume: 是否从上次训练的检查点恢复
        device: 设备选择 (例如: cpu, 0, 0,1,2,3)
        pretrained: 预训练模型路径或模型名称
        project: 保存结果的项目目录
        name: 实验名称
    """
    
    # 检查CUDA是否可用
    if not device:
        # 如果未指定device，自动检测
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"===== 训练设备: {device} =====")
    if device.startswith('cuda'):
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 确保数据文件存在
    if not os.path.exists(data):
        raise FileNotFoundError(f"数据配置文件未找到: {data}")
    
    # 创建结果目录
    # os.makedirs(os.path.join(project, name), exist_ok=True)
    
    # 加载模型
    print(f"===== 加载模型: {pretrained} =====")
    try:
        if pretrained.endswith('.pt') and os.path.exists(pretrained):
            # 加载本地模型文件
            model = YOLO(pretrained)
            print(f"从本地加载模型: {pretrained}")
        else:
            # 加载预训练模型
            model = YOLO(pretrained)
            print(f"加载预训练模型: {pretrained}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试从头开始训练...")
        model = YOLO('yolov8n.yaml')  # 从配置文件创建新模型
    
    # 开始训练
    print(f"===== 开始训练 =====")
    print(f"数据集配置: {data}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {img_size}")
    
    # 记录训练开始时间
    start_time = datetime.now()
    print(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 开始训练
    model.train(
        data=data,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        resume=resume,
        project=project,
        name=name,
    )
    
    # 记录训练结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总训练时间: {duration}")
    
    # 保存最终模型
    final_model_path = os.path.join(project, name, 'weights', 'best.pt')
    print(f"最佳模型保存在: {os.path.abspath(final_model_path)}")
    
    return model, final_model_path

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.abspath(os.path.join(script_dir, '..', 'mini_dataset', 'yolo.yaml'))

    print(f"使用的数据配置文件路径: {data_yaml}")

    
    train_model(
        data=data_yaml,
        epochs=10,
        batch_size=16,
        img_size=640,
        resume=False,
        device='',
        pretrained='yolov8n.pt',
        project='runs/train',
        name='exp'
    )

if __name__ == "__main__":
    main() 