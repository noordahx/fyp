import pytest
import torch
from torch.utils.data import TensorDataset
from mia_lib.data import create_subset_dataloader

def test_create_subset_dataloader():
    images = torch.rand(100, 3, 32, 32)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(images, labels)

    indices = list(range(5))
    batch_size = 2
    
    dataloader = create_subset_dataloader(
        dataset,
        indices,
        batch_size,
        shuffle=False,
        num_workers=0
    )

    data_iter = iter(dataloader)
    batch_images, batch_lables = next(data_iter)

    assert batch_images.shape == (batch_size, 3, 32, 32)
    assert batch_lables.shape == (batch_size,)

    assert len(dataloader.dataset) == len(indices)