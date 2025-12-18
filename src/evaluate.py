import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from .dataset import get_robustness_loader

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return acc, f1

def run_robustness_test(model, data_dir, device, batch_size=32):
    results = []
    
    # Clean Baseline
    print("Evaluating on Clean Test Set...")
    clean_loader = get_robustness_loader(data_dir, degradation=None, batch_size=batch_size)
    clean_acc, clean_f1 = evaluate_model(model, clean_loader, device)
    results.append({
        'Condition': 'Clean',
        'Parameter': 'None',
        'Accuracy': clean_acc,
        'F1-Score': clean_f1
    })
    
    # Gaussian Noise
    sigmas = [0.1, 0.5, 1.0]
    for sigma in sigmas:
        print(f"Evaluating Gaussian Noise (sigma={sigma})...")
        loader = get_robustness_loader(data_dir, degradation={'type': 'noise', 'params': {'std': sigma}}, batch_size=batch_size)
        acc, f1 = evaluate_model(model, loader, device)
        results.append({
            'Condition': 'Gaussian Noise',
            'Parameter': f'sigma={sigma}',
            'Accuracy': acc,
            'F1-Score': f1
        })
        
    # Blur
    kernels = [3, 5, 7]
    for k in kernels:
        print(f"Evaluating Blur (kernel={k})...")
        loader = get_robustness_loader(data_dir, degradation={'type': 'blur', 'params': {'kernel_size': k}}, batch_size=batch_size)
        acc, f1 = evaluate_model(model, loader, device)
        results.append({
            'Condition': 'Blur',
            'Parameter': f'k={k}',
            'Accuracy': acc,
            'F1-Score': f1
        })
        
    # Contrast
    factors = [0.3, 0.5, 0.8]
    for f in factors:
        print(f"Evaluating Contrast (factor={f})...")
        loader = get_robustness_loader(data_dir, degradation={'type': 'contrast', 'params': {'factor': f}}, batch_size=batch_size)
        acc, f1 = evaluate_model(model, loader, device)
        results.append({
            'Condition': 'Contrast',
            'Parameter': f'factor={f}',
            'Accuracy': acc,
            'F1-Score': f1
        })
        
    df = pd.DataFrame(results)
    
    # Calculate drops relative to clean
    df['Acc Drop'] = clean_acc - df['Accuracy']
    df['F1 Drop'] = clean_f1 - df['F1-Score']
    
    return df
