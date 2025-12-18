# Brain Tumor MRI Classification and Robustness Analysis

This project implements deep learning models for classifying brain tumor MRI images and analyzes their robustness against various image degradations (noise, blur, contrast changes).

## Project Overview

The goal is to classify MRI scans into four categories and evaluate how well different architectures perform under stress conditions (robustness testing).

### Classes
- **glioma**
- **meningioma**
- **no_tumor**
- **pituitary**

## Models Implemented

The project compares the following architectures:
1.  **Custom CNN (`custom_cnn`)**: A baseline 5-layer Convolutional Neural Network.
2.  **ResNet50 (`resnet50`)**: A standard deep residual network (pre-trained on ImageNet).
3.  **Vision Transformer (`vit_b_16`)**: A transformer-based architecture for vision tasks (pre-trained).
4.  **YOLOv8 Classification (`yolov8_cls`)**: Ultralytics' YOLOv8 model adapted for classification.

## Project Structure

```
.
├── classification_task/       # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── src/                       # Source code
│   ├── dataset.py             # Data loading and augmentation
│   ├── models.py              # Model definitions
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation and robustness testing
│   └── utils.py               # Utility functions
├── results/                   # Output directory for logs and CSVs
├── main.py                    # Main entry point for training
├── stress_test.ipynb          # Notebook for detailed robustness analysis
├── dataset.ipynb              # Notebook for downloading and preparing data
└── requirements.txt           # Python dependencies
```

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
Ensure your dataset is located in `classification_task/` with `train`, `val`, and `test` subdirectories. You can use `dataset.ipynb` to download the BRISC 2025 dataset and create the validation split.

### 2. Training & Evaluation
Run `main.py` to train a model and automatically run the robustness test on the test set.

```bash
python main.py --model <model_name> [options]
```

**Arguments:**
- `--model`: Model architecture to use. Choices: `custom_cnn`, `resnet50`, `vit_b_16`, `yolov8_cls`. (Required)
- `--data_dir`: Path to dataset (default: `./classification_task`).
- `--epochs`: Number of training epochs (default: 100).
- `--batch_size`: Batch size (default: 64).
- `--output_dir`: Directory to save results (default: `results`).

**Example:**
```bash
python main.py --model resnet50 --epochs 50 --batch_size 32
```

### 3. Robustness Analysis (Stress Test)
The `stress_test.ipynb` notebook allows you to load pre-trained models (`*_best.pt`) and perform a comprehensive stress test with varying levels of:
- Gaussian Noise
- Gaussian Blur
- Contrast Adjustment

## Results
- **Model Weights**: The best model weights are saved as `<model_name>_best.pt` in the root directory.
- **Logs**: Training logs are saved in the `results/` folder.
- **CSV Reports**: Robustness test results are saved as CSV files in the `results/` folder.
