import torch
import torch.nn as nn
import torchvision.models as models
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class CustomBaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomBaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024), # Assuming 224x224 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(model_name, num_classes, pretrained=True):
    if model_name == 'custom_cnn':
        model = CustomBaselineCNN(num_classes)
    
    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'yolov8_cls':
        if YOLO is None:
            raise ImportError("ultralytics package is required for YOLOv8. pip install ultralytics")
        # Load a pre-trained classification model (nano version by default for speed/size)
        # We load the weights then extract the pytorch module
        yolo = YOLO('yolov8n-cls.pt') 
        model = yolo.model
        # YOLOv8-cls head modification
        # The classification head is usually the last module. 
        # We need to ensure it outputs 'num_classes'.
        # Ultralytics classification head structure:
        # model.model[-1] is the Classify layer.
        # It has a linear layer.
        
        # However, simply replacing the linear layer might be tricky with their specific structure.
        # A safer bet for a custom loop is to re-initialize the final linear layer.
        # The Classify module has 'linear' attribute.
        
        classify_layer = model.model[-1]
        if hasattr(classify_layer, 'linear'):
            in_features = classify_layer.linear.in_features
            classify_layer.linear = nn.Linear(in_features, num_classes)
        
        # YOLO forward pass might return a tuple or object, we need to ensure it returns logits.
        # We might need to wrap it.
        
        class YOLOWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                output = self.model(x)
                if isinstance(output, (tuple, list)):
                    return output[0]
                return output
        
        model = YOLOWrapper(model)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    return model
