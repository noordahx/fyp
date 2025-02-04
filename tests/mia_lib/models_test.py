import pytest
from mia_lib.models import create_model

def test_create_model_resnet18():
    config = {
        "model": {
            "architecture": "resnet18",
            "pretrained": False,
            "num_classes": 10
        }
    }
    model = create_model(config)

    # Check final layer
    assert model.fc.out_features == config["model"]["num_classes"]

def test_create_model_resnet18_pretrained():
    config = {
        "model": {
            "architecture": "resnet18",
            "pretrained": True,
            "num_classes": 10
        }
    }
    model = create_model(config)
    # Check final layer
    assert model.fc.out_features == config["model"]["num_classes"]
