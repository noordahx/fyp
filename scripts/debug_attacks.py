#!/usr/bin/env python3
"""
Debug script for membership inference attacks.

This script trains a simple model and tests the attacks to identify issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mia_lib.attack import ThresholdAttack, LossBasedAttack
from sklearn.metrics import roc_auc_score, accuracy_score

def create_simple_model(num_classes=10):
    """Create a simple CNN model."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )

def train_simple_model(model, train_loader, device, epochs=5):
    """Train a simple model."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
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
            
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], '
                      f'Loss: {running_loss/100:.4f}, '
                      f'Accuracy: {100*correct/total:.2f}%')
                running_loss = 0.0

def analyze_model_behavior(model, train_loader, test_loader, device):
    """Analyze model behavior on training vs test data."""
    model.eval()
    
    train_confidences = []
    train_losses = []
    test_confidences = []
    test_losses = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        # Analyze training data
        count = 0
        for images, labels in train_loader:
            if count >= 1000:  # Limit for analysis
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            losses = criterion(outputs, labels)
            
            train_confidences.extend(max_probs.cpu().numpy())
            train_losses.extend(losses.cpu().numpy())
            count += len(images)
        
        # Analyze test data
        count = 0
        for images, labels in test_loader:
            if count >= 1000:  # Limit for analysis
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            losses = criterion(outputs, labels)
            
            test_confidences.extend(max_probs.cpu().numpy())
            test_losses.extend(losses.cpu().numpy())
            count += len(images)
    
    train_confidences = np.array(train_confidences)
    train_losses = np.array(train_losses)
    test_confidences = np.array(test_confidences)
    test_losses = np.array(test_losses)
    
    print("\n=== MODEL BEHAVIOR ANALYSIS ===")
    print(f"Training data:")
    print(f"  Confidence - Mean: {train_confidences.mean():.4f}, Std: {train_confidences.std():.4f}")
    print(f"  Loss - Mean: {train_losses.mean():.4f}, Std: {train_losses.std():.4f}")
    
    print(f"Test data:")
    print(f"  Confidence - Mean: {test_confidences.mean():.4f}, Std: {test_confidences.std():.4f}")
    print(f"  Loss - Mean: {test_losses.mean():.4f}, Std: {test_losses.std():.4f}")
    
    # Check if there's a meaningful difference
    conf_diff = train_confidences.mean() - test_confidences.mean()
    loss_diff = test_losses.mean() - train_losses.mean()
    
    print(f"\nDifferences:")
    print(f"  Confidence difference (train - test): {conf_diff:.4f}")
    print(f"  Loss difference (test - train): {loss_diff:.4f}")
    
    if abs(conf_diff) < 0.01 and abs(loss_diff) < 0.1:
        print("WARNING: Very small differences detected. Model may not be overfitting enough for MIA.")
    
    return {
        'train_confidences': train_confidences,
        'train_losses': train_losses,
        'test_confidences': test_confidences,
        'test_losses': test_losses
    }

def test_threshold_attack(model, train_loader, test_loader, device):
    """Test threshold attack with detailed analysis."""
    print("\n=== TESTING THRESHOLD ATTACK ===")
    
    attack = ThresholdAttack(device=device)
    
    # Use smaller subsets for calibration
    train_subset = DataLoader(
        torch.utils.data.Subset(train_loader.dataset, list(range(500))),
        batch_size=64, shuffle=False
    )
    test_subset = DataLoader(
        torch.utils.data.Subset(test_loader.dataset, list(range(500))),
        batch_size=64, shuffle=False
    )
    
    # Calibrate
    threshold = attack.calibrate_threshold(model, train_subset, test_subset)
    
    # Create evaluation dataset
    eval_data = []
    eval_labels = []
    
    # Add 500 training samples (members)
    count = 0
    for data, target in train_loader:
        if count >= 500:
            break
        eval_data.append(data)
        eval_labels.extend([1] * len(data))
        count += len(data)
    
    # Add 500 test samples (non-members)
    count = 0
    for data, target in test_loader:
        if count >= 500:
            break
        eval_data.append(data)
        eval_labels.extend([0] * len(data))
        count += len(data)
    
    eval_data = torch.cat(eval_data)
    eval_labels = torch.tensor(eval_labels)
    eval_dataset = TensorDataset(eval_data, eval_labels)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    # Run attack
    results = attack.infer_membership(model, eval_loader)
    
    print(f"Threshold Attack Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Threshold: {results['threshold']:.4f}")
    
    return results

def test_loss_attack(model, train_loader, test_loader, device):
    """Test loss attack with detailed analysis."""
    print("\n=== TESTING LOSS ATTACK ===")
    
    attack = LossBasedAttack(device=device)
    
    # Use smaller subsets for calibration
    train_subset = DataLoader(
        torch.utils.data.Subset(train_loader.dataset, list(range(500))),
        batch_size=64, shuffle=False
    )
    test_subset = DataLoader(
        torch.utils.data.Subset(test_loader.dataset, list(range(500))),
        batch_size=64, shuffle=False
    )
    
    # Calibrate
    threshold = attack.calibrate_threshold(model, train_subset, test_subset)
    
    # Create evaluation dataset with targets and membership labels
    eval_data = []
    eval_targets = []
    eval_membership_labels = []
    
    # Add 500 training samples (members)
    count = 0
    for data, target in train_loader:
        if count >= 500:
            break
        eval_data.append(data)
        eval_targets.append(target)
        eval_membership_labels.extend([1] * len(data))  # 1 = member
        count += len(data)
    
    # Add 500 test samples (non-members)
    count = 0
    for data, target in test_loader:
        if count >= 500:
            break
        eval_data.append(data)
        eval_targets.append(target)
        eval_membership_labels.extend([0] * len(data))  # 0 = non-member
        count += len(data)
    
    eval_data = torch.cat(eval_data)
    eval_targets = torch.cat(eval_targets)
    eval_membership_labels = torch.tensor(eval_membership_labels)
    
    # Create dataset with three elements: (data, class_targets, membership_labels)
    eval_dataset = TensorDataset(eval_data, eval_targets, eval_membership_labels)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    # Run attack
    results = attack.infer_membership(model, eval_loader)
    
    print(f"Loss Attack Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Threshold: {results['threshold']:.4f}")
    
    return results

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Use smaller datasets for faster debugging
    train_subset = torch.utils.data.Subset(train_dataset, list(range(5000)))
    test_subset = torch.utils.data.Subset(test_dataset, list(range(1000)))
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    # Create and train model
    model = create_simple_model().to(device)
    print("Training model...")
    train_simple_model(model, train_loader, device, epochs=10)
    
    # Analyze model behavior
    behavior = analyze_model_behavior(model, train_loader, test_loader, device)
    
    # Test attacks
    threshold_results = test_threshold_attack(model, train_loader, test_loader, device)
    loss_results = test_loss_attack(model, train_loader, test_loader, device)
    
    print("\n=== SUMMARY ===")
    print(f"Threshold Attack AUC: {threshold_results['auc']:.4f}")
    print(f"Loss Attack AUC: {loss_results['auc']:.4f}")
    
    if threshold_results['auc'] < 0.6 and loss_results['auc'] < 0.6:
        print("\nPOSSIBLE ISSUES:")
        print("1. Model is not overfitting enough")
        print("2. Dataset is too small")
        print("3. Model architecture is too simple")
        print("4. Training time is insufficient")
        print("\nSUGGESTIONS:")
        print("- Train for more epochs")
        print("- Use a larger model")
        print("- Reduce regularization")
        print("- Use more training data")

if __name__ == '__main__':
    main() 