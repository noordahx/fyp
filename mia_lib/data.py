# load dataset (CIFAR10) and return DataLoader

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_cifar10_dataloaders(CFG):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if CFG.target_model.use_data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = test_transform

    trainset = torchvision.datasets.CIFAR10(
        root=CFG.target_model.root,
        train=True,
        download=True,
        transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=CFG.target_model.root,
        train=False,
        download=True,
        transform=test_transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=CFG.target_model.train_batch_size,
        shuffle=True,
        num_workers=CFG.target_model.num_workers
    )

    testloader = DataLoader(
        testset,
        batch_size=CFG.target_model.eval_batch_size,
        shuffle=False,
        num_workers=CFG.target_model.num_workers
    )

    return trainset, testset, trainloader, testloader

def create_subset_dataloader(dataset, indices, batch_size, shuffle, num_workers):
    """
    Helper to create a DataLoader from a subset of a given dataset.
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )
    return loader