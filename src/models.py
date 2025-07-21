import torch.nn as nn
import torchvision.models as models

def _disable_inplace_operations(model):
    """
    Disable inplace operations in the model to make it compatible with Opacus DP-SGD.
    This fixes the common 'BackwardHookFunctionBackward is a view' error.
    """
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
        elif isinstance(module, nn.LeakyReLU):
            module.inplace = False
        elif isinstance(module, nn.ELU):
            module.inplace = False
        elif isinstance(module, nn.SELU):
            module.inplace = False
        elif isinstance(module, nn.GELU):
            # GELU doesn't have inplace parameter, but check for custom implementations
            pass
    return model

def create_model(model_config) -> nn.Module:
    model_name = model_config.architecture.lower()
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if model_config.pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, model_config.num_classes)
        
        # Fix inplace operations for DP compatibility
        model = _disable_inplace_operations(model)
        
    elif model_name == 'simple_cnn':
        # Simple CNN that's fully DP-compatible for testing
        model = _create_simple_cnn(model_config.num_classes)
        
    elif model_name == 'resnet18_dp':
        # DP-compatible ResNet18 without residual connections
        model = _create_dp_resnet18(model_config.num_classes)
        
    # Add other models here
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    return model

def _create_simple_cnn(num_classes):
    """
    Create a simple CNN that's fully compatible with DP-SGD.
    No residual connections, BatchNorm, or inplace operations.
    """
    return nn.Sequential(
        # First block
        nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(4, 32),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2),
        
        # Second block  
        nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, 64),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2),
        
        # Third block
        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), 
        nn.GroupNorm(16, 128),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2),
        
        # Classifier
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 256),
        nn.ReLU(inplace=False),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

def _create_dp_resnet18(num_classes):
    """
    Create a ResNet18-like architecture without residual connections for DP compatibility.
    Similar depth and structure to ResNet18 but purely feedforward.
    """
    return nn.Sequential(
        # Initial conv layer (like ResNet18)
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.GroupNorm(32, 64),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        
        # Layer 1 blocks (64 channels)
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.GroupNorm(32, 64),
        nn.ReLU(inplace=False),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.GroupNorm(32, 64),
        nn.ReLU(inplace=False),
        
        # Layer 2 blocks (128 channels)  
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.GroupNorm(32, 128),
        nn.ReLU(inplace=False),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.GroupNorm(32, 128),
        nn.ReLU(inplace=False),
        
        # Layer 3 blocks (256 channels)
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
        nn.GroupNorm(32, 256),
        nn.ReLU(inplace=False),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.GroupNorm(32, 256),
        nn.ReLU(inplace=False),
        
        # Layer 4 blocks (512 channels)
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
        nn.GroupNorm(32, 512),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.GroupNorm(32, 512),
        nn.ReLU(inplace=False),
        
        # Global average pooling and classifier
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )
