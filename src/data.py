import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset(dataset_name: str, data_dir: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if dataset_name.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_dataset, test_dataset
