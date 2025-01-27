# factory methods to instantiate PyTorck models

import importlib
import torch.nn as nn
from torchvision.models import ResNet18_Weights

def create_model(config):
    """
    Create a model given the config. For example, a resnet18 from torchvision.
    """
    model_architecture = importlib.import_module(f"torchvision.models")
    model_class = getattr(model_architecture, config["model"]["architecture"])
    model = model_class(weights=ResNet18_Weights.DEFAULT) if config["model"]["pretrained"] else model_class()

    # Adjust the final layer to number of classes
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, config["model"]["num_classes"])
    
    # if classifier (e.g. MobileNet) that names the final layer differently
    elif hasattr(model, "classifier"):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, config["model"]["num_classes"])
    
    return model