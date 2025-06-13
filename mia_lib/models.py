# factory methods to instantiate PyTorch models

import importlib
import torch.nn as nn
from torchvision.models import ResNet18_Weights
import torchvision.models as models


def create_model(num_classes, pretrained=False):
    """
    Create a ResNet18 model for the given number of classes.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        PyTorch model
    """
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