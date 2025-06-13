# load dataset (CIFAR10, MNIST, CIFAR100) and return DataLoader

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def get_dataset(dataset_name, data_dir='./data'):
    """
    Get dataset by name. Returns train_dataset, test_dataset, num_classes.
    
    Args:
        dataset_name: 'cifar10', 'mnist', or 'cifar100'
        data_dir: directory to store/load data
    
    Returns:
        tuple: (train_dataset, test_dataset, num_classes)
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_datasets(data_dir)
    elif dataset_name.lower() == 'mnist':
        return get_mnist_datasets(data_dir)
    elif dataset_name.lower() == 'cifar100':
        return get_cifar100_datasets(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_cifar10_datasets(data_dir='./data'):
    """Get CIFAR-10 datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    return trainset, testset, 10


def get_mnist_datasets(data_dir='./data'):
    """Get MNIST datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    return trainset, testset, 10


def get_cifar100_datasets(data_dir='./data'):
    """Get CIFAR-100 datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    return trainset, testset, 100


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