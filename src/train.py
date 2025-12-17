import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import os
from .utils import EarlyStopping, calculate_metrics

def train_model(model, train_loader, val_loader, device, model_name, epochs=50, lr=1e-4, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler: Cosine Annealing Warm Restarts
    # T_0 is number of iterations for the first restart. Let's set it to 10 epochs worth of steps? 
    # Or just T_0=10 epochs.
    # User didn't specify T_0, so I'll pick a reasonable default, e.g., 10.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    
    scaler = GradScaler('cuda')
    
    # DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    checkpoint_path = f'{model_name}_best.pt'
    early_stopping = EarlyStopping(patience=7, verbose=True, path=checkpoint_path)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        # Step scheduler at epoch level (CosineAnnealingWarmRestarts can be stepped per batch or epoch, usually epoch is fine or T_0 is in epochs)
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc, train_f1 = calculate_metrics(torch.tensor(train_preds).unsqueeze(1), torch.tensor(train_targets)) # Mocking output format for utils
        # Actually utils expects outputs (logits) but I passed preds. Let's fix utils usage or just calc here.
        # My utils.calculate_metrics expects outputs (logits).
        # Let's just use sklearn directly here for simplicity or fix utils usage.
        # I'll just use the preds I collected.
        from sklearn.metrics import accuracy_score, f1_score
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        print(f'Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f}')
        print(f'Epoch {epoch+1}: Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}')
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    # Load best model
    if torch.cuda.device_count() > 1:
        # If wrapped, we need to unwrap to load state dict if we saved module.state_dict()
        # My EarlyStopping saves module.state_dict() if DataParallel.
        # So we can load it into model.module
        model.module.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path))
        
    return model
