#!/usr/bin/env python3
"""
Train models optimized for high MIA AUC on GPU servers.

This script is designed to run on GPU servers and create models that are
highly vulnerable to membership inference attacks for research purposes.

Usage:
    python scripts/train_high_auc_model.py --dataset cifar10 --epochs 100 --batch-size 256
    python scripts/train_high_auc_model.py --dataset mnist --model-size large --train-size 10000
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import sys
import json
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mia_lib.attack import ThresholdAttack, LossBasedAttack
from mia_lib.visualization import MIAVisualizer
from torch.utils.data import TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score

def create_overfitting_model(dataset_name: str, model_size: str = 'medium', num_classes: int = 10):
    """Create models designed to overfit for high MIA AUC."""
    
    if dataset_name.lower() == 'mnist':
        if model_size == 'small':
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        elif model_size == 'medium':
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        else:  # large
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
    
    else:  # CIFAR-10/100
        if model_size == 'small':
            return nn.Sequential(
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
                
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        elif model_size == 'medium':
            return nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        else:  # large
            return nn.Sequential(
                nn.Conv2d(3, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Flatten(),
                nn.Linear(512 * 4 * 4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )

def get_dataset(dataset_name: str, train_size: int = None, test_size: int = None):
    """Load dataset with optional size limits."""
    
    if dataset_name.lower() == 'cifar10':
        # Minimal augmentation to encourage overfitting
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced augmentation
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
        num_classes = 10
        
    elif dataset_name.lower() == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        num_classes = 100
        
    else:  # MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        num_classes = 10
    
    # Limit dataset sizes if specified
    if train_size and train_size < len(train_dataset):
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
    
    if test_size and test_size < len(test_dataset):
        test_indices = np.random.choice(len(test_dataset), test_size, replace=False)
        test_dataset = Subset(test_dataset, test_indices)
    
    return train_dataset, test_dataset, num_classes

def train_model(model, train_loader, test_loader, device, epochs, lr, weight_decay=0):
    """Train model with overfitting-friendly settings."""
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler to help with convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    
    best_train_acc = 0.0
    best_overfitting_gap = 0.0
    
    print(f"Training for {epochs} epochs on {device}")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
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
        train_loss = running_loss / len(train_loader)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        
        # Test phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_loss = test_loss / len(test_loader)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        # Update learning rate
        scheduler.step(train_acc)
        
        overfitting_gap = train_acc - test_acc
        
        # Track best overfitting
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if overfitting_gap > best_overfitting_gap:
            best_overfitting_gap = overfitting_gap
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch < 5:
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch+1}/{epochs}] ({elapsed:.1f}s)')
            print(f'  Train: Acc={train_acc:.2f}%, Loss={train_loss:.4f}')
            print(f'  Test:  Acc={test_acc:.2f}%, Loss={test_loss:.4f}')
            print(f'  Gap:   {overfitting_gap:.2f}%')
            print(f'  LR:    {optimizer.param_groups[0]["lr"]:.6f}')
    
    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    print(f"Best train accuracy: {best_train_acc:.2f}%")
    print(f"Best overfitting gap: {best_overfitting_gap:.2f}%")
    
    return {
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_train_acc': best_train_acc,
        'best_overfitting_gap': best_overfitting_gap
    }

def evaluate_mia_attacks(model, train_dataset, test_dataset, device, eval_size=2000):
    """Evaluate MIA attacks on the trained model."""
    
    print(f"\nEvaluating MIA attacks with {eval_size} samples each...")
    
    # Create evaluation subsets
    train_eval_size = min(eval_size, len(train_dataset))
    test_eval_size = min(eval_size, len(test_dataset))
    
    train_indices = np.random.choice(len(train_dataset), train_eval_size, replace=False)
    test_indices = np.random.choice(len(test_dataset), test_eval_size, replace=False)
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    results = {}
    
    # Threshold Attack
    print("Running Threshold Attack...")
    threshold_attack = ThresholdAttack(device=device)
    
    # Use smaller calibration sets
    calib_size = min(500, len(train_subset) // 2)
    train_calib = DataLoader(Subset(train_subset, list(range(calib_size))), batch_size=64, shuffle=False)
    test_calib = DataLoader(Subset(test_subset, list(range(calib_size))), batch_size=64, shuffle=False)
    
    threshold_attack.calibrate_threshold(model, train_calib, test_calib)
    
    # Create evaluation dataset
    eval_data = []
    eval_labels = []
    
    for data, target in train_loader:
        eval_data.append(data)
        eval_labels.extend([1] * len(data))
    
    for data, target in test_loader:
        eval_data.append(data)
        eval_labels.extend([0] * len(data))
    
    eval_data = torch.cat(eval_data)
    eval_labels = torch.tensor(eval_labels)
    eval_dataset = TensorDataset(eval_data, eval_labels)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    
    threshold_results = threshold_attack.infer_membership(model, eval_loader)
    results['threshold'] = threshold_results
    
    # Loss Attack
    print("Running Loss Attack...")
    loss_attack = LossBasedAttack(device=device)
    loss_attack.calibrate_threshold(model, train_calib, test_calib)
    
    # Create evaluation dataset with targets for loss calculation
    eval_data = []
    eval_targets = []
    eval_labels = []
    
    for data, target in train_loader:
        eval_data.append(data)
        eval_targets.append(target)
        eval_labels.extend([1] * len(data))
    
    for data, target in test_loader:
        eval_data.append(data)
        eval_targets.append(target)
        eval_labels.extend([0] * len(data))
    
    eval_data = torch.cat(eval_data)
    eval_targets = torch.cat(eval_targets)
    eval_labels = torch.tensor(eval_labels)
    eval_dataset = TensorDataset(eval_data, eval_targets, eval_labels)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    
    loss_results = loss_attack.infer_membership(model, eval_loader)
    results['loss'] = loss_results
    
    # Print results
    print(f"\nMIA Results:")
    print(f"Threshold Attack - Accuracy: {threshold_results['accuracy']:.4f}, AUC: {threshold_results['auc']:.4f}")
    print(f"Loss Attack      - Accuracy: {loss_results['accuracy']:.4f}, AUC: {loss_results['auc']:.4f}")
    
    return results

def make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-compatible formats."""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description='Train high-AUC models for MIA research')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'mnist'],
                       help='Dataset to use')
    parser.add_argument('--train-size', type=int, default=None,
                       help='Limit training set size (default: use full dataset)')
    parser.add_argument('--test-size', type=int, default=None,
                       help='Limit test set size (default: use full dataset)')
    
    # Model arguments
    parser.add_argument('--model-size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Model size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                       help='Weight decay (0 for overfitting)')
    
    # Evaluation arguments
    parser.add_argument('--eval-size', type=int, default=2000,
                       help='Number of samples for MIA evaluation')
    parser.add_argument('--skip-mia', action='store_true',
                       help='Skip MIA evaluation')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./results/high_auc',
                       help='Output directory')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name (auto-generated if not provided)')
    
    # Visualization arguments
    parser.add_argument('--generate-report', action='store_true', default=True,
                       help='Generate visual reports (default: True)')
    parser.add_argument('--skip-report', action='store_true',
                       help='Skip visual report generation')
    
    args = parser.parse_args()
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate model name
    if args.model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{args.dataset}_{args.model_size}_{timestamp}"
    else:
        model_name = args.model_name
    
    print(f"Model name: {model_name}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, test_dataset, num_classes = get_dataset(
        args.dataset, args.train_size, args.test_size
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {num_classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True if device == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    model = create_overfitting_model(args.dataset, args.model_size, num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Train model
    training_start_time = time.time()
    training_history = train_model(
        model, train_loader, test_loader, device, 
        args.epochs, args.lr, args.weight_decay
    )
    total_training_time = time.time() - training_start_time
    
    # Save model
    model_path = output_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate MIA attacks
    mia_results = {}
    if not args.skip_mia:
        mia_results = evaluate_mia_attacks(
            model, train_dataset, test_dataset, device, args.eval_size
        )
    
    # Save results
    results = {
        'model_name': model_name,
        'dataset': args.dataset,
        'model_size': args.model_size,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'device': device,
        'total_params': total_params,
        'training_history': training_history,
        'mia_results': mia_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Make results JSON serializable
    results_serializable = make_json_serializable(results)
    
    results_path = output_dir / f"{model_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Generate visual reports
    if args.generate_report and not args.skip_report and mia_results:
        print(f"\n{'='*60}")
        print(f"GENERATING VISUAL REPORTS")
        print(f"{'='*60}")
        
        try:
            # Create visualizer
            viz_dir = output_dir / "visualizations"
            visualizer = MIAVisualizer(viz_dir)
            
            # Prepare model info for visualization
            model_info = {
                'model_name': model_name,
                'dataset': args.dataset,
                'model_size': args.model_size,
                'total_params': total_params,
                'train_size': len(train_dataset),
                'test_size': len(test_dataset),
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'device': device,
                'training_time': f"{total_training_time:.1f}s"
            }
            
            # Generate individual reports
            print("Generating training analysis report...")
            training_report = visualizer.create_training_report(
                training_history, model_info, f"{model_name}_training"
            )
            print(f"✓ Training report saved: {training_report}")
            
            print("Generating MIA attack analysis report...")
            mia_report = visualizer.create_mia_report(
                mia_results, model_info, f"{model_name}_mia"
            )
            print(f"✓ MIA report saved: {mia_report}")
            
            print("Generating executive summary dashboard...")
            dashboard = visualizer.create_summary_dashboard(
                training_history, mia_results, model_info, f"{model_name}_dashboard"
            )
            print(f"✓ Dashboard saved: {dashboard}")
            
            print("Generating comprehensive HTML report...")
            html_report = visualizer.generate_html_report(
                training_history, mia_results, model_info, save_name=model_name
            )
            print(f"✓ HTML report saved: {html_report}")
            
            print(f"\n🎉 All visual reports generated successfully!")
            print(f"📁 Reports location: {viz_dir}")
            print(f"🌐 Open HTML report: {html_report}")
            
        except Exception as e:
            print(f"⚠️  Error generating visual reports: {e}")
            print("Continuing without visualization...")
    
    # Print summary
    final_train_acc = training_history['train_accuracies'][-1]
    final_test_acc = training_history['test_accuracies'][-1]
    overfitting_gap = final_train_acc - final_test_acc
    
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset} ({len(train_dataset)} train, {len(test_dataset)} test)")
    print(f"Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Overfitting Gap: {overfitting_gap:.2f}%")
    
    if mia_results:
        print(f"\nMIA ATTACK RESULTS:")
        for attack_name, attack_results in mia_results.items():
            print(f"{attack_name.capitalize()} Attack:")
            print(f"  Accuracy: {attack_results['accuracy']:.4f}")
            print(f"  AUC: {attack_results['auc']:.4f}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS FOR HIGH AUC:")
    print(f"{'='*60}")
    
    if overfitting_gap > 30:
        print("✓ Excellent overfitting achieved!")
    elif overfitting_gap > 15:
        print("✓ Good overfitting. Consider training longer or reducing regularization.")
    else:
        print("⚠ Low overfitting. Try:")
        print("  - Increase epochs")
        print("  - Reduce training data size")
        print("  - Use larger model")
        print("  - Remove regularization")
    
    if mia_results:
        best_auc = max(result['auc'] for result in mia_results.values())
        if best_auc > 0.8:
            print("✓ Excellent MIA vulnerability!")
        elif best_auc > 0.7:
            print("✓ Good MIA vulnerability.")
        elif best_auc > 0.6:
            print("~ Moderate MIA vulnerability. Consider more overfitting.")
        else:
            print("✗ Low MIA vulnerability. Model needs more overfitting.")

if __name__ == '__main__':
    main() 