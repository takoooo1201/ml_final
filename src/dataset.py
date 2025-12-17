import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Standard ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_transforms(split='train', degradation=None):
    """
    degradation: dict with keys 'type' and 'params'
    e.g. {'type': 'noise', 'params': {'std': 0.1}}
    """
    transform_list = []
    
    # Resize is constant
    transform_list.append(transforms.Resize((224, 224)))
    
    if split == 'train':
        # Add some standard augmentation for training if desired, 
        # but user didn't explicitly ask for it besides the robustness test.
        # We'll keep it simple or add basic flips.
        transform_list.append(transforms.RandomHorizontalFlip())
    
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(MEAN, STD))
    
    # Apply degradations AFTER normalization (or before? Noise is usually additive to pixel values or normalized values)
    # If we add noise to normalized tensor, the sigma needs to be scaled or interpreted relative to normalized range.
    # Usually easier to add noise to normalized tensor for consistency.
    
    if degradation:
        deg_type = degradation.get('type')
        params = degradation.get('params', {})
        
        if deg_type == 'noise':
            # Gaussian Noise
            std = params.get('std', 0.1)
            transform_list.append(AddGaussianNoise(0., std))
            
        elif deg_type == 'blur':
            # Gaussian Blur
            kernel_size = params.get('kernel_size', 3)
            # Sigma is usually derived from kernel size or set. 
            # User only specified kernel size. We can set sigma to auto or something reasonable.
            transform_list.append(transforms.GaussianBlur(kernel_size=kernel_size))
            
        elif deg_type == 'contrast':
            # Contrast
            factor = params.get('factor', 1.0)
            # AdjustContrast expects PIL or Tensor. 
            # Since we already converted to Tensor and Normalized, we need to be careful.
            # ColorJitter/AdjustContrast works on Tensors.
            # However, it expects [0, 1] or [0, 255] usually? 
            # Documentation says "If input is a tensor, it is expected to be in [..., 1 or 3, H, W]".
            # It works on normalized tensors but the effect might be different.
            # To be safe, let's apply contrast BEFORE normalization.
            
            # Re-ordering list
            # Remove ToTensor and Normalize
            transform_list.pop() # Normalize
            transform_list.pop() # ToTensor
            
            # Add Contrast
            # factor 0.5 means lower contrast.
            transform_list.append(transforms.ColorJitter(contrast=(factor, factor)))
            
            # Add back ToTensor and Normalize
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize(MEAN, STD))
            
    return transforms.Compose(transform_list)

class SubsetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Check if train and test exist
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Load Test
        test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms('test'))
        num_classes = len(test_dataset.classes)
        
        if os.path.exists(val_dir):
            train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms('train'))
            val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms('val'))
        else:
            # Split train into train/val (80/20)
            full_train_dataset = datasets.ImageFolder(train_dir)
            train_size = int(0.8 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            
            # Use fixed seed for reproducibility
            generator = torch.Generator().manual_seed(42)
            train_ds, val_ds = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=generator)
            
            train_dataset = SubsetWrapper(train_ds, get_transforms('train'))
            val_dataset = SubsetWrapper(val_ds, get_transforms('val'))
            
    elif not os.path.exists(train_dir):
        # Fallback for flat directory
        full_dataset = datasets.ImageFolder(data_dir)
        
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
        
        train_dataset = SubsetWrapper(train_ds, get_transforms('train'))
        val_dataset = SubsetWrapper(val_ds, get_transforms('val'))
        test_dataset = SubsetWrapper(test_ds, get_transforms('test'))
        
        num_classes = len(full_dataset.classes)
        
    else:
        # Fallback if only train exists but no test? 
        train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms('train'))
        val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms('val')) if os.path.exists(val_dir) else None
        test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms('test')) if os.path.exists(test_dir) else None
        num_classes = len(train_dataset.classes)

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset) if val_dataset else 0}, Test={len(test_dataset) if test_dataset else 0}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, num_classes

def get_robustness_loader(data_dir, degradation, batch_size=32, num_workers=4):
    # Similar logic but for test set with degradation
    test_dir = os.path.join(data_dir, 'test')
    
    transform = get_transforms('test', degradation=degradation)
    
    if not os.path.exists(test_dir):
        # Fallback logic (re-split? This is risky if random seed changes. 
        # Ideally we save the split. For this script, we'll assume the user provides a split or we use the same seed.)
        # To ensure consistency, we should probably rely on the user providing a proper dataset structure.
        # But let's try to be robust.
        
        full_dataset = datasets.ImageFolder(data_dir)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        # MUST use same seed if we want same test set
        generator = torch.Generator().manual_seed(42)
        _, _, test_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
        
        class SubsetWrapper(Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform
            def __getitem__(self, index):
                x, y = self.subset[index]
                if self.transform:
                    x = self.transform(x)
                return x, y
            def __len__(self):
                return len(self.subset)
                
        test_dataset = SubsetWrapper(test_ds, transform)
    else:
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
