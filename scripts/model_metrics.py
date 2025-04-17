#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate
import yaml
import json

def evaluate_model(
    model,
    data,
    img_size=640,
    batch_size=16,
    conf_thres=0.25,
    iou_thres=0.45,
    device='',
    save_dir='runs/eval',
    plot=True,
    verbose=True
):
    """
    评估YOLOv8模型性能，计算各项指标
    
    参数:
        model: 已加载的YOLO模型实例
        data: 数据集配置文件路径
        img_size: 图像尺寸
        batch_size: 批次大小
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
        device: 设备选择 (例如: cpu, 0, 0,1,2,3)
        save_dir: 保存评估结果的目录
        plot: 是否绘制评估结果图表
        verbose: 是否打印详细信息
    
    返回:
        metrics_dict: 包含各项评估指标的字典
        figures: 生成的图形路径列表
    """
    # 确保数据文件存在
    if not os.path.exists(data):
        raise FileNotFoundError(f"数据配置文件未找到: {data}")
    
    # 创建结果目录
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载类别信息
    try:
        with open(data, 'r') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config.get('names', [])
    except Exception as e:
        print(f"无法解析数据配置文件: {e}")
        class_names = model.names
    
    print("\n===== 开始模型评估 =====")
    
    # 进行验证
    metrics = model.val(
        data=data,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        conf=conf_thres,
        iou=iou_thres,
        save_json=True,
        save_hybrid=True,
        save_conf=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        plots=plot,
        verbose=verbose
    )
    
    # 提取指标
    metrics_dict = {}
    for k, v in metrics.results_dict.items():
        if isinstance(v, (int, float, np.int64, np.float64)):
            metrics_dict[k] = float(v)
    
    # 主要指标
    main_metrics = {k: v for k, v in metrics_dict.items() 
                  if k in ['metrics/precision(B)', 'metrics/recall(B)', 
                          'metrics/mAP50(B)', 'metrics/mAP50-95(B)']}
    
    # 打印表格形式的主要指标
    headers = ["指标", "值"]
    table_data = [
        ["精确率 (Precision)", f"{main_metrics.get('metrics/precision(B)', 0):.4f}"],
        ["召回率 (Recall)", f"{main_metrics.get('metrics/recall(B)', 0):.4f}"],
        ["mAP@0.5", f"{main_metrics.get('metrics/mAP50(B)', 0):.4f}"],
        ["mAP@0.5:0.95", f"{main_metrics.get('metrics/mAP50-95(B)', 0):.4f}"]
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 保存评估指标到CSV和JSON文件
    csv_path = save_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        f.write("metric,value\n")
        for k, v in metrics_dict.items():
            f.write(f"{k},{v}\n")
    
    json_path = save_dir / "metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"评估指标已保存到: {csv_path} 和 {json_path}")
    
    # 生成自定义图表
    figures = []
    
    # 1. 精确率-召回率曲线
    if plot:
        pr_curve_path = save_dir / "precision_recall_curve.png"
        if not (save_dir / "PR_curve.png").exists():  # 如果YOLOv8没有自动生成
            plt.figure(figsize=(10, 8))
            plt.title('精确率-召回率曲线')
            plt.xlabel('召回率')
            plt.ylabel('精确率')
            plt.grid(True)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            
            precision = main_metrics.get('metrics/precision(B)', 0.8)
            recall = main_metrics.get('metrics/recall(B)', 0.7)
            
            plt.scatter(recall, precision, s=100, c='red', marker='o', 
                       label=f'模型性能 (P={precision:.2f}, R={recall:.2f})')
            plt.legend()
            
            plt.savefig(pr_curve_path)
            plt.close()
            print(f"精确率-召回率曲线已保存到: {pr_curve_path}")
            figures.append(pr_curve_path)
    
    # 2. 性能总结图
    summary_path = save_dir / "performance_summary.png"
    if plot:
        plt.figure(figsize=(12, 6))
        
        # 绘制条形图
        metrics_labels = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
        metrics_values = [
            main_metrics.get('metrics/precision(B)', 0),
            main_metrics.get('metrics/recall(B)', 0),
            main_metrics.get('metrics/mAP50(B)', 0),
            main_metrics.get('metrics/mAP50-95(B)', 0)
        ]
        
        plt.bar(metrics_labels, metrics_values, color=['blue', 'green', 'orange', 'red'])
        plt.ylim([0, 1.0])
        plt.title('模型性能指标总结')
        plt.ylabel('指标值')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在柱形上方添加具体数值
        for i, v in enumerate(metrics_values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(summary_path)
        plt.close()
        print(f"性能总结图已保存到: {summary_path}")
        figures.append(summary_path)
    
    return metrics_dict, figures

def format_metrics_report(metrics_dict):
    """
    将指标数据格式化为人类可读的报告文本
    
    参数:
        metrics_dict: 包含各项评估指标的字典
    
    返回:
        report_text: 格式化的报告文本
    """
    # 格式化主要指标
    precision = metrics_dict.get('metrics/precision(B)', 0)
    recall = metrics_dict.get('metrics/recall(B)', 0)
    map50 = metrics_dict.get('metrics/mAP50(B)', 0)
    map50_95 = metrics_dict.get('metrics/mAP50-95(B)', 0)
    
    # 构建报告
    report = [
        "===== 模型评估报告 =====",
        "",
        f"精确率 (Precision): {precision:.4f}",
        f"召回率 (Recall): {recall:.4f}",
        f"mAP@0.5: {map50:.4f}",
        f"mAP@0.5:0.95: {map50_95:.4f}",
        "",
        "性能解读:",
    ]
    
    # 添加性能解读
    if map50 > 0.8:
        report.append("- 模型在检测目标位置方面表现优秀")
    elif map50 > 0.6:
        report.append("- 模型在检测目标位置方面表现良好")
    else:
        report.append("- 模型在检测目标位置方面有待提高")
    
    if map50_95 > 0.6:
        report.append("- 模型在定位精度方面表现优秀")
    elif map50_95 > 0.4:
        report.append("- 模型在定位精度方面表现良好")
    else:
        report.append("- 模型在定位精度方面有待提高")
    
    if precision > recall:
        report.append("- 模型更注重预测的准确性，可能存在漏检现象")
    else:
        report.append("- 模型更注重检出率，可能存在误检现象")
    
    # F1分数
    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        report.append(f"- F1分数: {f1:.4f}")
    
    return "\n".join(report)

if __name__ == "__main__":
    # 简单测试函数
    print("模型评估指标模块已加载") 