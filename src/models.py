import torch.nn as nn
import torchvision.models as models

def create_model(model_config) -> nn.Module:
    model_name = model_config.architecture.lower()
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if model_config.pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, model_config.num_classes)
    # Add other models here
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    return model
