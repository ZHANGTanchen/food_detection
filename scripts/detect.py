#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import torch
import os
import cv2
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from model_metrics import evaluate_model, format_metrics_report

def detect_objects(
    weights='runs/train/exp/weights/best.pt',
    source='../dataset/images/test',
    img_size=640,
    conf_thres=0.25,
    iou_thres=0.45,
    device='',
    save_dir='runs/detect',
    save_txt=False,
    show=False,
    classes=None,
    evaluate=False,
    data=None,
    batch_size=16,
    plot_metrics=True
):
    """
    使用YOLOv8进行目标检测，同时可选择性评估模型性能
    
    参数:
        weights: 模型权重文件路径
        source: 要检测的图像或视频路径或目录
        img_size: 图像尺寸
        conf_thres: 检测置信度阈值
        iou_thres: NMS IOU 阈值
        device: 设备选择 (例如: cpu, 0, 0,1,2,3)
        save_dir: 保存结果的目录
        save_txt: 是否保存结果为文本文件
        show: 是否在屏幕上显示结果
        classes: 仅检测指定类别
        evaluate: 是否评估模型性能
        data: 数据集配置文件路径，用于评估
        batch_size: 评估时的批次大小
        plot_metrics: 是否绘制评估指标图表
    
    返回:
        results: 检测结果
        metrics: 评估指标（如果evaluate=True）
    """
    
    # 检查CUDA是否可用
    if not device:
        # 如果未指定device，自动检测
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"===== 检测设备: {device} =====")
    if device.startswith('cuda'):
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 确保模型文件存在
    if not os.path.exists(weights):
        raise FileNotFoundError(f"模型文件未找到: {weights}")
    
    # 确保源文件或目录存在
    if not os.path.exists(source):
        raise FileNotFoundError(f"源文件或目录未找到: {source}")
    
    # 如果需要评估但未提供数据集配置文件，报错
    if evaluate and not data:
        raise ValueError("评估模型性能需要提供data参数（数据集配置文件路径）")
    
    # 创建结果目录结构
    save_dir = Path(save_dir)
    detect_dir = save_dir / 'detect'
    metrics_dir = save_dir / 'metrics'
    
    # os.makedirs(detect_dir, exist_ok=True)
    if evaluate:
        os.makedirs(metrics_dir, exist_ok=True)
    
    # 加载模型
    print(f"===== 加载模型: {weights} =====")
    model = YOLO(weights)
    
    # 开始检测
    print(f"===== 开始检测 =====")
    print(f"检测源: {source}")
    print(f"图像尺寸: {img_size}")
    print(f"置信度阈值: {conf_thres}")
    print(f"IOU阈值: {iou_thres}")
    
    # 记录检测开始时间
    start_time = time.time()
    
    # 进行检测
    results = model.predict(
        source=source,
        imgsz=img_size,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        save=True,
        save_txt=save_txt,
        save_conf=True,
        classes=classes,
        project=detect_dir.parent,
        name=detect_dir.name
    )
    
    # 显示结果
    if show:
        for result in results:
            img = result.orig_img
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            
            # 在图像上绘制检测结果
            for box, conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box.astype(int)
                label = f"{model.names[int(cls_id)]} {conf:.2f}"
                color = (0, 255, 0)  # 绿色边框
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 显示图像
            cv2.imshow("Detection Result", img)
            cv2.waitKey(0)
    
    # 记录检测结束时间并计算总耗时
    end_time = time.time()
    duration = end_time - start_time
    print(f"检测完成！总耗时: {duration:.2f} 秒")
    print(f"检测结果保存在: {os.path.abspath(detect_dir)}")
    
    # 如果需要，评估模型性能
    metrics_data = None
    figures = None
    
    if evaluate:
        print("\n===== 开始评估模型性能 =====")
        metrics_data, figures = evaluate_model(
            model=model,
            data=data,
            img_size=img_size,
            batch_size=batch_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            device=device,
            save_dir=metrics_dir,
            plot=plot_metrics,
            verbose=True
        )
        
        # 生成并显示性能报告文本
        report_text = format_metrics_report(metrics_data)
        print("\n" + report_text)
        
        # 将报告保存到文本文件
        report_path = metrics_dir / "performance_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n性能评估报告保存在: {os.path.abspath(report_path)}")
        print(f"性能评估结果保存在: {os.path.abspath(metrics_dir)}")
    
    # 根据是否评估返回不同的结果
    if evaluate:
        return results, metrics_data, figures
    else:
        return results

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.abspath(os.path.join(script_dir, '..', 'mini_dataset', 'images', 'test'))
    data_yaml = os.path.abspath(os.path.join(script_dir, '..', 'mini_dataset', 'yolo.yaml'))

    print(f"使用的数据配置文件路径: {data_yaml}")
    print(f"使用的源目录路径: {source_dir}")

    detect_objects(
        weights='runs/train/exp/weights/best.pt',
        source=source_dir,
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
        device='',
        save_dir='runs/detect_with_eval',
        save_txt=True,
        show=False,
        classes=None,
        # 模型评估相关参数
        evaluate=True,
        data=data_yaml,
        batch_size=16,
        plot_metrics=True
    )

if __name__ == "__main__":
    main() 