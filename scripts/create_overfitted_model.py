#!/usr/bin/env python3
"""
Create an overfitted model for testing membership inference attacks.

This script trains a model that intentionally overfits to demonstrate
effective membership inference attacks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def create_overfitting_model(num_classes=10):
    """Create a model designed to overfit."""
    return nn.Sequential(
        # Simpler but still capable model
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        # No dropout to encourage overfitting
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

def train_overfitted_model(model, train_loader, test_loader, device, epochs=30):
    """Train a model to overfit."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    # Use reasonable learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_accuracies.append(train_acc)
        
        # Test phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accuracies.append(test_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Accuracy: {train_acc:.2f}%')
            print(f'  Test Accuracy: {test_acc:.2f}%')
            print(f'  Overfitting Gap: {train_acc - test_acc:.2f}%')
    
    return train_accuracies, test_accuracies

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load CIFAR-10 with minimal augmentation to encourage overfitting
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Use smaller training set to encourage overfitting
    train_subset = Subset(train_dataset, list(range(5000)))  # Only 5000 samples
    test_subset = Subset(test_dataset, list(range(1000)))    # 1000 test samples
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # Create and train model
    model = create_overfitting_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Training overfitted model...")
    train_accs, test_accs = train_overfitted_model(model, train_loader, test_loader, device, epochs=30)
    
    # Save the overfitted model
    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / 'overfitted_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Overfitted model saved to {model_path}")
    
    # Print final statistics
    final_train_acc = train_accs[-1]
    final_test_acc = test_accs[-1]
    overfitting_gap = final_train_acc - final_test_acc
    
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {final_train_acc:.2f}%")
    print(f"  Test Accuracy: {final_test_acc:.2f}%")
    print(f"  Overfitting Gap: {overfitting_gap:.2f}%")
    
    if overfitting_gap > 20:
        print("✓ Good overfitting achieved! This model should work well for MIA.")
    elif overfitting_gap > 10:
        print("~ Moderate overfitting. MIA might work but not optimally.")
    else:
        print("✗ Insufficient overfitting. MIA will likely fail.")
    
    # Test the model with our attacks
    print("\nTesting attacks on overfitted model...")
    
    # Import and test attacks
    from mia_lib.attack import ThresholdAttack, LossBasedAttack
    from torch.utils.data import TensorDataset
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Test threshold attack
    attack = ThresholdAttack(device=device)
    
    # Use small subsets for quick testing
    train_test_subset = DataLoader(Subset(train_subset, list(range(500))), batch_size=64, shuffle=False)
    test_test_subset = DataLoader(Subset(test_subset, list(range(500))), batch_size=64, shuffle=False)
    
    attack.calibrate_threshold(model, train_test_subset, test_test_subset)
    
    # Create evaluation data
    eval_data = []
    eval_labels = []
    
    # Add training samples (members)
    for data, target in train_test_subset:
        eval_data.append(data)
        eval_labels.extend([1] * len(data))
    
    # Add test samples (non-members)
    for data, target in test_test_subset:
        eval_data.append(data)
        eval_labels.extend([0] * len(data))
    
    eval_data = torch.cat(eval_data)
    eval_labels = torch.tensor(eval_labels)
    eval_dataset = TensorDataset(eval_data, eval_labels)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    results = attack.infer_membership(model, eval_loader)
    
    print(f"Threshold Attack Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    
    if results['auc'] > 0.7:
        print("✓ Excellent attack performance!")
    elif results['auc'] > 0.6:
        print("✓ Good attack performance!")
    elif results['auc'] > 0.55:
        print("~ Moderate attack performance.")
    else:
        print("✗ Poor attack performance.")

if __name__ == '__main__':
    main() 