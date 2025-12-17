import argparse
import torch
import os
import sys
import datetime
import pandas as pd
from src.models import get_model
from src.dataset import get_dataloaders
from src.train import train_model
from src.evaluate import run_robustness_test

class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    parser = argparse.ArgumentParser(description='Brain Tumor MRI Classification and Robustness Analysis')
    parser.add_argument('--model', type=str, required=True, choices=['custom_cnn', 'resnet50', 'vit_b_16', 'yolov8_cls'], help='Model architecture to use')
    parser.add_argument('--data_dir', type=str, default='./classification_task', help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Global batch size')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    log_file = os.path.join(args.output_dir, f'{args.model}_training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    sys.stdout = Logger(log_file)
    print(f"Logging to {log_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    print(f"Number of classes: {num_classes}")
    
    # Model
    print(f"Initializing {args.model}...")
    model = get_model(args.model, num_classes=num_classes)
    
    # Train
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, device, args.model, epochs=args.epochs)
    
    # Robustness Test
    print("Starting robustness test...")
    df = run_robustness_test(model, args.data_dir, device, batch_size=args.batch_size)
    
    # Save results
    csv_path = os.path.join(args.output_dir, f'{args.model}_robustness_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    print(df)

if __name__ == '__main__':
    main()
