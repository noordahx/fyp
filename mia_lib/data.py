# load dataset (CIFAR10) and return DataLoader

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_cifar10_dataloaders(config):
    """
    Returns train and test DataLoaers for CIFAR-10,
    or optinoally returns subsets for target/shadow training.
    """
    transform = transforms.Compose([
        transforms.Resize((config["dataset"]["input_size"], config["dataset"]["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=config["dataset"]["root"],
        train=True,
        download=config["dataset"]["download"],
        transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=config["dataset"]["root"],
        train=False,
        download=config["dataset"]["download"],
        transform=transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=config["dataset"]["train_batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"]
    )

    testloader = DataLoader(
        testset,
        batch_size=config["dataset"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"]
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
        num_workers=num_workers
    )
    return loader