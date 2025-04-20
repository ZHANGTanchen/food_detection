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
    Object detection using YOLOv8 with optional model evaluation
    
    Parameters:
        weights: Path to model weights file
        source: Path or directory of images or videos to detect
        img_size: Image size
        conf_thres: Detection confidence threshold
        iou_thres: NMS IOU threshold
        device: Device selection (e.g., cpu, 0, 0,1,2,3)
        save_dir: Directory to save results
        save_txt: Whether to save results as text files
        show: Whether to display results on screen
        classes: Only detect specific classes
        evaluate: Whether to evaluate model performance
        data: Dataset configuration file path for evaluation
        batch_size: Batch size for evaluation
        plot_metrics: Whether to plot evaluation metrics
    
    Returns:
        results: Detection results
        metrics: Evaluation metrics (if evaluate=True)
    """
    
    # Check if CUDA is available
    if not device:
        # Auto-detect device if not specified
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"===== Detection Device: {device} =====")
    if device.startswith('cuda'):
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Ensure model file exists
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model file not found: {weights}")
    
    # Ensure source file or directory exists
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file or directory not found: {source}")
    
    # Raise error if evaluation is enabled but no data configuration file is provided
    if evaluate and not data:
        raise ValueError("Data configuration file path (data parameter) is required for model evaluation")
    
    # Create result directory structure
    save_dir = Path(save_dir)
    detect_dir = save_dir / 'detect'
    metrics_dir = save_dir / 'metrics'
    
    # os.makedirs(detect_dir, exist_ok=True)
    if evaluate:
        os.makedirs(metrics_dir, exist_ok=True)
    
    # Load model
    print(f"===== Loading Model: {weights} =====")
    model = YOLO(weights)
    
    # Start detection
    print(f"===== Starting Detection =====")
    print(f"Detection Source: {source}")
    print(f"Image Size: {img_size}")
    print(f"Confidence Threshold: {conf_thres}")
    print(f"IOU Threshold: {iou_thres}")
    
    # Record detection start time
    start_time = time.time()
    
    # Perform detection
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
    
    # Display results
    if show:
        for result in results:
            img = result.orig_img
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            
            # Draw detection results on image
            for box, conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box.astype(int)
                label = f"{model.names[int(cls_id)]} {conf:.2f}"
                color = (0, 255, 0)  # Green border
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show image
            cv2.imshow("Detection Result", img)
            cv2.waitKey(0)
    
    # Record detection end time and calculate total duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"Detection completed! Total time: {duration:.2f} seconds")
    print(f"Detection results saved to: {os.path.abspath(detect_dir)}")
    
    # Evaluate model performance if needed
    metrics_data = None
    figures = None
    
    if evaluate:
        print("\n===== Starting Model Evaluation =====")
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
        
        # Generate and display performance report text
        report_text = format_metrics_report(metrics_data)
        print("\n" + report_text)
        
        # Save report to text file
        report_path = metrics_dir / "performance_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\nPerformance report saved to: {os.path.abspath(report_path)}")
        print(f"Evaluation results saved to: {os.path.abspath(metrics_dir)}")
    
    # Return different results based on whether evaluation was performed
    if evaluate:
        return results, metrics_data, figures
    else:
        return results

def main():
    # Get script directory and build absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'images', 'test'))
    data_yaml = os.path.abspath(os.path.join(script_dir, '..', 'data', 'yolo.yaml'))

    print(f"Using data configuration file: {data_yaml}")
    print(f"Using source directory: {source_dir}")

    # Call detection function with appropriate parameters
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
        # Model evaluation parameters
        evaluate=True,
        data=data_yaml,
        batch_size=16,
        plot_metrics=True
    )

if __name__ == "__main__":
    main() 