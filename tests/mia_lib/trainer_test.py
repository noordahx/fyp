import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mia_lib.trainer import train_model

class DummyModel(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.fc = nn.Linear(32*32*3, num_classes)
    
    def forward(self, x):
        # Flatten input except for batch dimension
        x = x.view(x.size(0), -1)
        return self.fc(x)

@pytest.fixture
def dummy_loaders():
    images = torch.rand(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)

    train_subset, val_subset = torch.utils.data.random_split(dataset, [8, 8])
    train_loader = DataLoader(train_subset, batch_size=10)
    val_loader = DataLoader(val_subset, batch_size=10)
    return train_loader, val_loader

@pytest.fixture
def dummy_config(tmp_path):
    return {
        "training": {
            "lr": 0.001,
            "weight_decay": 0.0,
            "epochs": 2,
            "early_stop_patience": 3
        }
    }, tmp_path / "model.pth"

def test_train_model(dummy_loaders, dummy_config):
    (config, save_path) = dummy_config
    train_loader, val_loader = dummy_loaders
    model = DummyModel(num_classes=10)
    device = torch.device("cpu")

    best_acc, best_loss = train_model(model, train_loader, val_loader, config, device, save_path)

    assert 0 <= best_acc <= 1, "Best accuracy should be between 0 and 1"
    assert best_loss >= 0, "Best loss should be non-negative"
    assert save_path.exists(), "Model should be saved to disk"