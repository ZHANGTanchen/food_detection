# Food Detection with YOLOv8

This repository contains a food detection system built using YOLOv8, designed to identify various food items in images.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Detection](#detection)
- [Evaluation](#evaluation)
- [Creating a Mini Dataset](#creating-a-mini-dataset)

## Installation

### Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

For users in regions with slow connections to PyPI, you can use a local mirror:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Using Conda (Optional)

For a cleaner environment, you can use Conda:

```bash
# Create a new conda environment
conda create -n yolov8_env python=3.8
conda activate yolov8_env

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The dataset should be organized in the following structure:

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── yolo.yaml
```

The `yolo.yaml` file should contain:

```yaml
path: "."  # Path relative to the yaml file
train: images/train
val: images/valid
test: images/test

nc: 6  # Number of classes
names: ['Onion', 'beef', 'chicken', 'eggs', 'potato', 'radish']
```

## Training

The `train.py` script allows you to train a YOLOv8 model on your dataset.

### Usage

```bash
python scripts/train.py
```

### Parameters

The main function in `train.py` uses these parameters (which can be modified directly in the code):

```python
train_model(
    data='../dataset/yolo.yaml',  # Dataset configuration file
    epochs=50,                    # Number of training epochs
    batch_size=16,                # Batch size
    img_size=640,                 # Image size
    resume=False,                 # Resume training from the last checkpoint
    device='',                    # Device to use (empty string for auto-detection)
    pretrained=True,              # Use pretrained weights
    project='runs/train',         # Project name for saving results
    name='exp'                    # Experiment name
)
```

### Features

- Automatic CUDA detection for GPU acceleration
- Model checkpoint saving
- Training progress visualization
- Comprehensive logging

## Detection

The `detect.py` script allows you to run inference on images or videos and evaluate the model.

### Usage

```bash
python scripts/detect.py
```

### Parameters

The main parameters for the detect function are:

```python
detect_objects(
    weights='runs/train/exp/weights/best.pt',  # Model weights
    source='../dataset/images/test',           # Source directory or file
    img_size=640,                              # Image size
    conf_thres=0.25,                           # Confidence threshold
    iou_thres=0.45,                            # IoU threshold
    device='',                                 # Device to use (empty string for auto-detection)
    save_dir='runs/detect_with_eval',          # Directory to save results
    save_txt=False,                            # Save text results
    show=False,                                # Show results on screen
    classes=None,                              # Filter by class
    evaluate=True,                             # Evaluate model performance
    data='../dataset/yolo.yaml',               # Dataset configuration (for evaluation)
    batch_size=16,                             # Batch size for evaluation
    plot_metrics=True                          # Plot evaluation metrics
)
```

### Features

- Object detection with bounding boxes
- Optional result visualization
- Performance evaluation
- Metrics saving in CSV and text format
- Precision-recall curve plotting

## Evaluation

The model evaluation is integrated into the detection process when you set `evaluate=True`. This feature:

1. Evaluates model performance on your test set
2. Calculates precision, recall, and mAP metrics
3. Generates visual plots for analysis
4. Saves detailed performance reports

## License

[MIT License](LICENSE) 