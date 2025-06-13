#!/usr/bin/env python3
"""
Comprehensive Training Script with Differential Privacy

This script provides a command-line interface for training models with various
differential privacy methods and evaluating their effectiveness against 
membership inference attacks.

Usage:
    python scripts/train_with_dp.py --method dp_sgd_opacus --dataset cifar10 --epsilon 1.0
    python scripts/train_with_dp.py --method pate --dataset mnist --num-teachers 5
    python scripts/train_with_dp.py --method standard --dataset cifar10  # baseline
"""

import argparse
import logging
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.dp_training import DPTrainer, create_teacher_data_splits
from mia_lib.models import create_model
from mia_lib.attack import ShadowAttack, ThresholdAttack, LossBasedAttack

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_dataset(dataset_name: str, data_dir: str = './data'):
    """
    Load and return train/test datasets.
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'mnist', 'cifar100')
        data_dir: Directory to store/load data
        
    Returns:
        Tuple of (train_dataset, test_dataset, num_classes)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    if dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        num_classes = 10
        
    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        num_classes = 10
        
    elif dataset_name.lower() == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        num_classes = 100
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes


def create_simple_model(dataset_name: str, num_classes: int) -> nn.Module:
    """
    Create a simple model for the given dataset.
    
    Args:
        dataset_name: Name of dataset
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    if dataset_name.lower() == 'mnist':
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    else:  # CIFAR-10/100
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
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )


def plot_training_metrics(history, save_dir, model_name):
    """Plot and save training metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics - {model_name}', fontsize=16)
    
    # Plot training and validation loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Training Loss', marker='o')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', marker='o')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Plot training and validation accuracy
    if 'train_acc' in history and 'val_acc' in history:
        axes[0, 1].plot(history['train_acc'], label='Training Accuracy', marker='o')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', marker='o')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot learning rate if available
    if 'lr' in history:
        axes[1, 0].plot(history['lr'], label='Learning Rate', marker='o')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot privacy budget if available
    if 'epsilon' in history:
        axes[1, 1].plot(history['epsilon'], label='Privacy Budget (ε)', marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Privacy Budget Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'training_metrics_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training metrics to {save_dir / f'training_metrics_{model_name}.png'}")


def plot_privacy_utility_tradeoff(results, save_dir):
    """Plot and save privacy-utility tradeoff curves."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Privacy-Utility Tradeoff Analysis', fontsize=16)
    
    # Extract data
    epsilons = sorted([float(k) for k in results.keys() if k != 'standard'])
    model_accuracies = [results[str(eps)]['model_accuracy'] for eps in epsilons]
    attack_accuracies = [results[str(eps)]['attack_accuracy'] for eps in epsilons]
    attack_aucs = [results[str(eps)]['attack_auc'] for eps in epsilons]
    
    # Add standard model if available
    if 'standard' in results:
        epsilons.append(float('inf'))
        model_accuracies.append(results['standard']['model_accuracy'])
        attack_accuracies.append(results['standard']['attack_accuracy'])
        attack_aucs.append(results['standard']['attack_auc'])
    
    # Plot model accuracy vs epsilon
    ax1.semilogx(epsilons, model_accuracies, 'o-', label='Model Accuracy')
    ax1.set_xlabel('Privacy Budget (ε)')
    ax1.set_ylabel('Model Accuracy')
    ax1.set_title('Model Performance vs Privacy Budget')
    ax1.grid(True)
    ax1.legend()
    
    # Plot attack metrics vs epsilon
    ax2.semilogx(epsilons, attack_accuracies, 'o-', label='Attack Accuracy')
    ax2.semilogx(epsilons, attack_aucs, 's-', label='Attack AUC')
    ax2.set_xlabel('Privacy Budget (ε)')
    ax2.set_ylabel('Attack Success Rate')
    ax2.set_title('Attack Performance vs Privacy Budget')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved privacy-utility tradeoff to {save_dir / 'privacy_utility_tradeoff.png'}")


def plot_attack_metrics_comparison(results, save_dir):
    """Plot and save attack metrics comparison across different privacy levels."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    data = []
    for method, method_results in results.items():
        epsilon = method if method != 'standard' else 'inf'
        data.append({
            'Method': f'ε={epsilon}',
            'Model Accuracy': method_results['model_accuracy'],
            'Attack Accuracy': method_results['attack_accuracy'],
            'Attack AUC': method_results['attack_auc']
        })
    
    df = pd.DataFrame(data)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Performance Comparison Across Privacy Levels', fontsize=16)
    
    # Model accuracy
    sns.barplot(data=df, x='Method', y='Model Accuracy', ax=axes[0])
    axes[0].set_title('Model Accuracy')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Attack accuracy
    sns.barplot(data=df, x='Method', y='Attack Accuracy', ax=axes[1])
    axes[1].set_title('Attack Accuracy')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Attack AUC
    sns.barplot(data=df, x='Method', y='Attack AUC', ax=axes[2])
    axes[2].set_title('Attack AUC')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'attack_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved attack metrics comparison to {save_dir / 'attack_metrics_comparison.png'}")


def create_dp_visualizations(training_history, attack_results, output_dir, model_name):
    """Create and save all DP-related visualizations."""
    logger.info("Creating DP visualizations...")
    
    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training metrics
    if training_history:
        plot_training_metrics(training_history, viz_dir, model_name)
    
    # Plot privacy-utility tradeoff if we have multiple epsilon values
    if len(attack_results) > 1:
        plot_privacy_utility_tradeoff(attack_results, viz_dir)
        plot_attack_metrics_comparison(attack_results, viz_dir)
    
    logger.info(f"DP visualizations saved to {viz_dir}")


def run_membership_inference_attacks(model: nn.Module, train_loader: DataLoader,
                                   test_loader: DataLoader, device: str,
                                   output_dir: Path) -> dict:
    """
    Run membership inference attacks on the trained model.
    
    Args:
        model: Trained model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to run on
        output_dir: Directory to save results
        
    Returns:
        Dictionary with attack results
    """
    logger.info("Running membership inference attacks...")
    
    results = {}
    
    # Threshold Attack
    logger.info("Running Threshold Attack...")
    threshold_attack = ThresholdAttack(device=device)
    threshold_attack.calibrate_threshold(model, train_loader, test_loader)
    
    # Create combined loader for testing
    combined_data = []
    combined_labels = []
    combined_targets = []
    
    # Add training samples (members)
    for data, target in train_loader:
        combined_data.append(data)
        combined_targets.append(target)
        combined_labels.extend([1] * len(data))  # 1 = member
        
    # Add test samples (non-members)  
    for data, target in test_loader:
        combined_data.append(data)
        combined_targets.append(target)
        combined_labels.extend([0] * len(data))  # 0 = non-member
        
    combined_data = torch.cat(combined_data)
    combined_targets = torch.cat(combined_targets)
    combined_labels = torch.tensor(combined_labels)
    combined_dataset = torch.utils.data.TensorDataset(combined_data, combined_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
    
    threshold_results = threshold_attack.infer_membership(model, combined_loader)
    results['threshold_attack'] = threshold_results
    
    # Loss-based Attack
    logger.info("Running Loss-based Attack...")
    loss_attack = LossBasedAttack(device=device)
    loss_attack.calibrate_threshold(model, train_loader, test_loader)
    
    # Create combined dataset with targets for loss calculation
    combined_dataset_with_targets = torch.utils.data.TensorDataset(combined_data, combined_targets)
    combined_loader_with_targets = DataLoader(combined_dataset_with_targets, batch_size=128, shuffle=False)
    
    loss_results = loss_attack.infer_membership(model, combined_loader_with_targets)
    results['loss_attack'] = loss_results
    
    # Calculate summary metrics
    summary_results = {}
    for attack_name, attack_results in results.items():
        if 'accuracy' in attack_results:
            summary_results[f'{attack_name}_accuracy'] = attack_results['accuracy']
        if 'auc' in attack_results:
            summary_results[f'{attack_name}_auc'] = attack_results['auc']
    
    # Use the best attack result for summary
    best_attack_acc = max([summary_results.get(f'{name}_accuracy', 0) for name in ['threshold_attack', 'loss_attack']])
    best_attack_auc = max([summary_results.get(f'{name}_auc', 0.5) for name in ['threshold_attack', 'loss_attack']])
    
    summary_results['attack_accuracy'] = best_attack_acc
    summary_results['attack_auc'] = best_attack_auc
    
    # Save detailed results
    results_file = output_dir / 'attack_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for attack_name, attack_results in results.items():
            serializable_results[attack_name] = {}
            for key, value in attack_results.items():
                if hasattr(value, 'tolist'):
                    serializable_results[attack_name][key] = value.tolist()
                else:
                    serializable_results[attack_name][key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Attack results saved to {results_file}")
    
    # Print summary
    for attack_name, attack_results in results.items():
        if 'accuracy' in attack_results:
            logger.info(f"{attack_name} accuracy: {attack_results['accuracy']:.4f}")
    
    return summary_results


def main():
    parser = argparse.ArgumentParser(description='Train models with differential privacy')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to store/load data')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    
    # Training arguments
    parser.add_argument('--method', type=str, default='standard',
                       choices=['standard', 'dp_sgd_custom', 
                               'pate', 'output_perturbation'],
                       help='Training method to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    
    # Privacy arguments
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Privacy budget epsilon')
    parser.add_argument('--delta', type=float, default=1e-5,
                       help='Privacy budget delta')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--noise-multiplier', type=float, default=1.0,
                       help='Noise multiplier for DP-SGD')
    
    # PATE arguments
    parser.add_argument('--num-teachers', type=int, default=5,
                       help='Number of teacher models for PATE')
    parser.add_argument('--teacher-epochs', type=int, default=10,
                       help='Number of epochs for teacher training')
    parser.add_argument('--student-epochs', type=int, default=10,
                       help='Number of epochs for student training')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Name for saved model (auto-generated if not provided)')
    parser.add_argument('--run-attacks', action='store_true',
                       help='Run membership inference attacks after training')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generating visualizations')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate model name if not provided
    if args.model_name is None:
        model_name = f"{args.dataset}_{args.method}_eps{args.epsilon}"
        if args.method == 'pate':
            model_name += f"_teachers{args.num_teachers}"
    else:
        model_name = args.model_name
    
    save_path = output_dir / f"{model_name}.pt"
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    train_dataset, test_dataset, num_classes = get_dataset(args.dataset, args.data_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create model
    model = create_simple_model(args.dataset, num_classes)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = DPTrainer(model, device)
    
    # Train model based on method
    logger.info(f"Training with method: {args.method}")
    
    if args.method == 'standard':
        history = trainer.train_standard(
            train_loader, test_loader, args.epochs, args.lr, str(save_path)
        )
        
    elif args.method == 'dp_sgd_opacus':
        history = trainer.train_with_dp_sgd_opacus(
            train_loader, test_loader, args.epochs, args.lr,
            args.epsilon, args.delta, args.max_grad_norm, str(save_path)
        )
        
    elif args.method == 'dp_sgd_custom':
        history = trainer.train_with_dp_sgd_custom(
            train_loader, test_loader, args.epochs, args.lr,
            args.epsilon, args.delta, args.max_grad_norm, 
            args.noise_multiplier, str(save_path)
        )
        
    elif args.method == 'pate':
        # Create teacher data splits
        teacher_loaders = create_teacher_data_splits(train_dataset, args.num_teachers)
        
        history = trainer.train_with_pate(
            teacher_loaders, train_loader, test_loader,
            args.teacher_epochs, args.student_epochs,
            args.epsilon, args.delta, str(save_path)
        )
        
    elif args.method == 'output_perturbation':
        history = trainer.train_with_output_perturbation(
            train_loader, test_loader, args.epochs, args.lr,
            args.epsilon, args.delta, sensitivity=1.0, save_path=str(save_path)
        )
        
    else:
        raise ValueError(f"Unknown training method: {args.method}")
    
    logger.info("Training completed!")
    
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    model_accuracy = correct / total
    logger.info(f"Final model accuracy: {model_accuracy:.4f}")
    
    # Run membership inference attacks if requested
    attack_results = {}
    if args.run_attacks:
        attack_summary = run_membership_inference_attacks(
            model, train_loader, test_loader, device, output_dir
        )
        attack_summary['model_accuracy'] = model_accuracy
        attack_results[str(args.epsilon) if args.method != 'standard' else 'standard'] = attack_summary
    
    # Save training history
    history_file = output_dir / f"{model_name}_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate visualizations
    if not args.no_visualizations:
        create_dp_visualizations(history, attack_results, output_dir, model_name)
    
    # Save final summary
    summary = {
        'method': args.method,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'epsilon': args.epsilon if args.method != 'standard' else None,
        'delta': args.delta if args.method != 'standard' else None,
        'final_test_accuracy': history.get('test_acc', [])[-1] if 'test_acc' in history else model_accuracy,
        'model_path': str(save_path),
        'device': device
    }
    
    if args.run_attacks and attack_results:
        summary.update(attack_results[list(attack_results.keys())[0]])
    
    summary_file = output_dir / f"{model_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved to {summary_file}")
    logger.info(f"Model saved to {save_path}")


if __name__ == '__main__':
    main() 