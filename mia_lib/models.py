# factory methods to instantiate PyTorck models

import importlib
import torch.nn as nn
from torchvision.models import ResNet18_Weights

def create_model(CFG):
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