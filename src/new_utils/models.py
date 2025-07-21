# factory methods to instantiate PyTorch models

import importlib
import torch.nn as nn
from torchvision.models import ResNet18_Weights
import torchvision.models as models


def create_model(config_or_num_classes, pretrained=False, input_channels=3):
    """
    Unified model creation function that accepts either:
    - A config object with model.num_classes, model.pretrained attributes
    - Direct num_classes, pretrained, input_channels parameters
    
    Returns a PyTorch model
    """
    # Check if first argument is a config object or num_classes
    if isinstance(config_or_num_classes, int):
        num_classes = config_or_num_classes
    else:
        # It's a config object
        model_config = config_or_num_classes
        if hasattr(model_config, 'model'):
            # Full config object
            num_classes = model_config.model.num_classes
            pretrained = model_config.model.pretrained
            # Input channels based on dataset
            if hasattr(model_config, 'data') and model_config.data.dataset.lower() == 'mnist':
                input_channels = 1
        else:
            # Direct model config
            num_classes = model_config.num_classes
            pretrained = getattr(model_config, 'pretrained', False)
    
    if input_channels == 1:
        # Simple CNN for MNIST (grayscale)
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    else:
        # ResNet18 for RGB images
        if pretrained:
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=None)
        
        # Adjust the final layer to number of classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        return model


def create_model_from_config(CFG):
    """
    Create a model given the config. For example, a resnet18 from torchvision.
    """
    model_architecture = importlib.import_module(f"torchvision.models")
    model_class = getattr(model_architecture, CFG.target_model.architecture)
    model = model_class(weights=ResNet18_Weights.DEFAULT) if CFG.target_model.pretrained else model_class()

    # Adjust the final layer to number of classes
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(CFG.target_model.dropout_rate),
            nn.Linear(in_features, CFG.target_model.num_classes)
        )

    return model