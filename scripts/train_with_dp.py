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
    python scripts/train_with_dp.py --method output_perturbation --dataset cifar10 --epsilon 1.0
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

from mia_lib.dp_training import DPTrainer, create_teacher_data_splits
from mia_lib.data import get_dataset

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
    
    # Create trainer with appropriate architecture for dataset
    input_channels = 1 if args.dataset == 'mnist' else 3
    trainer = DPTrainer(num_classes, device, pretrained=False, input_channels=input_channels)
    logger.info(f"Created model with {sum(p.numel() for p in trainer.model.parameters())} parameters")
    
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
    trainer.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = trainer.model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    model_accuracy = correct / total
    logger.info(f"Final model accuracy: {model_accuracy:.4f}")
    
    # Save training history
    history_file = output_dir / f"{model_name}_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate visualizations
    if not args.no_visualizations:
        plot_training_metrics(history, output_dir, model_name)
    
    # Save final summary
    # Get final test accuracy from history
    final_test_acc = model_accuracy
    if 'test_acc' in history and len(history['test_acc']) > 0:
        final_test_acc = history['test_acc'][-1]
    elif 'val_acc' in history and len(history['val_acc']) > 0:
        final_test_acc = history['val_acc'][-1]
    
    summary = {
        'method': args.method,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'epsilon': args.epsilon if args.method != 'standard' else None,
        'delta': args.delta if args.method != 'standard' else None,
        'final_test_accuracy': final_test_acc,
        'model_path': str(save_path),
        'device': device
    }
    
    summary_file = output_dir / f"{model_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved to {summary_file}")
    logger.info(f"Model saved to {save_path}")


if __name__ == '__main__':
    main() 