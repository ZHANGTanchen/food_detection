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
    Train a YOLOv8 model
    
    Parameters:
        data: Path to dataset configuration file
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        resume: Whether to resume from last checkpoint
        device: Device selection (e.g., cpu, 0, 0,1,2,3)
        pretrained: Path or name of pretrained model
        project: Directory to save results
        name: Experiment name
    """
    
    # Check if CUDA is available
    if not device:
        # Auto-detect device if not specified
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"===== Training Device: {device} =====")
    if device.startswith('cuda'):
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Ensure data file exists
    if not os.path.exists(data):
        raise FileNotFoundError(f"Data configuration file not found: {data}")
    
    # Create results directory
    # os.makedirs(os.path.join(project, name), exist_ok=True)
    
    # Load model
    print(f"===== Loading Model: {pretrained} =====")
    try:
        if pretrained.endswith('.pt') and os.path.exists(pretrained):
            # Load local model file
            model = YOLO(pretrained)
            print(f"Loading model from local file: {pretrained}")
        else:
            # Load pretrained model
            model = YOLO(pretrained)
            print(f"Loading pretrained model: {pretrained}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Attempting to train from scratch...")
        model = YOLO('yolov8n.yaml')  # Create new model from configuration file
    
    # Start training
    print(f"===== Starting Training =====")
    print(f"Dataset configuration: {data}")
    print(f"Training epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    
    # Record training start time
    start_time = datetime.now()
    print(f"Training start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Start training
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
    
    # Record training end time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Training end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {duration}")
    
    # Save final model
    final_model_path = os.path.join(project, name, 'weights', 'best.pt')
    print(f"Best model saved to: {os.path.abspath(final_model_path)}")
    
    return model, final_model_path

def main():
    # Get script directory and build absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.abspath(os.path.join(script_dir, '..', 'data', 'yolo.yaml'))

    print(f"Using data configuration file: {data_yaml}")
    
    train_model(
        data=data_yaml,
        epochs=1,
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