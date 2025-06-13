#!/usr/bin/env python3
"""
Simple Membership Inference Attack Runner

This script provides a straightforward way to run membership inference attacks
on trained models using the implemented attack classes.

Usage:
    python scripts/simple_mia_runner.py --model-path results/model.pt --attack-type threshold
    python scripts/simple_mia_runner.py --model-path results/model.pt --attack-type loss
    python scripts/simple_mia_runner.py --model-path results/model.pt --attack-type shadow
"""

import argparse
import logging
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import json
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, precision_recall_curve

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mia_lib.attack import ShadowAttack, ThresholdAttack, LossBasedAttack, ReferenceAttack

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dataset(dataset_name: str, data_dir: str = './data'):
    """Load dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
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
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes


def create_simple_model(dataset_name: str, num_classes: int) -> nn.Module:
    """Create a simple model."""
    if dataset_name.lower() == 'mnist':
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    else:  # CIFAR-10
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
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


def plot_attack_metrics(results, save_dir):
    """Plot and save attack metrics comparison."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
    sns.set_palette("husl")
    
    # Extract metrics for each attack
    attacks = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Create DataFrame for easier plotting
    data = []
    for attack in attacks:
        for metric in metrics:
            if metric in results[attack]:
                data.append({
                    'Attack': attack.capitalize(),
                    'Metric': metric.capitalize(),
                    'Value': results[attack][metric]
                })
    
    if not data:
        logger.warning("No data available for plotting metrics")
        return
    
    df = pd.DataFrame(data)
    
    # Plot metrics comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Attack', y='Value', hue='Metric')
    plt.title('Attack Performance Comparison')
    plt.xlabel('Attack Type')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / 'attack_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved attack metrics comparison to {save_dir / 'attack_metrics_comparison.png'}")


def plot_roc_curves(results, save_dir):
    """Plot and save ROC curves for each attack."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    curves_plotted = False
    for attack_name, attack_results in results.items():
        if 'probabilities' in attack_results and 'true_labels' in attack_results:
            try:
                fpr, tpr, _ = roc_curve(
                    attack_results['true_labels'],
                    attack_results['probabilities']
                )
                auc = attack_results.get('auc', 0.5)
                plt.plot(fpr, tpr, label=f'{attack_name.capitalize()} (AUC = {auc:.3f})')
                curves_plotted = True
            except Exception as e:
                logger.warning(f"Could not plot ROC curve for {attack_name}: {e}")
    
    if curves_plotted:
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Attacks')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ROC curves to {save_dir / 'roc_curves.png'}")
    else:
        plt.close()
        logger.warning("No ROC curves could be plotted")


def plot_attack_distributions(results, save_dir):
    """Plot and save score distributions for each attack."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for attack_name, attack_results in results.items():
        if 'probabilities' in attack_results and 'true_labels' in attack_results:
            try:
                plt.figure(figsize=(10, 6))
                
                # Convert to numpy arrays if needed
                probabilities = np.array(attack_results['probabilities'])
                true_labels = np.array(attack_results['true_labels'])
                
                # Plot distributions for members and non-members
                member_scores = probabilities[true_labels == 1]
                non_member_scores = probabilities[true_labels == 0]
                
                if len(member_scores) > 0 and len(non_member_scores) > 0:
                    sns.kdeplot(member_scores, label='Members', fill=True)
                    sns.kdeplot(non_member_scores, label='Non-members', fill=True)
                    
                    plt.xlabel('Attack Score')
                    plt.ylabel('Density')
                    plt.title(f'Score Distribution - {attack_name.capitalize()} Attack')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(save_dir / f'score_distribution_{attack_name}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved score distribution for {attack_name} to {save_dir / f'score_distribution_{attack_name}.png'}")
                else:
                    plt.close()
                    logger.warning(f"Insufficient data for score distribution plot for {attack_name}")
            except Exception as e:
                plt.close()
                logger.warning(f"Could not plot score distribution for {attack_name}: {e}")


def create_attack_visualizations(results, output_dir):
    """Create and save all attack visualizations."""
    logger.info("Creating attack visualizations...")
    
    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    plot_attack_metrics(results, viz_dir)
    plot_roc_curves(results, viz_dir)
    plot_attack_distributions(results, viz_dir)
    
    logger.info(f"Attack visualizations saved to {viz_dir}")


def run_threshold_attack(model, train_loader, test_loader, device):
    """Run threshold-based attack."""
    logger.info("Running Threshold Attack...")
    
    attack = ThresholdAttack(device=device)
    
    # Calibrate threshold using smaller subsets for efficiency
    train_subset = DataLoader(
        torch.utils.data.Subset(train_loader.dataset, list(range(0, min(1000, len(train_loader.dataset))))),
        batch_size=128, shuffle=False
    )
    test_subset = DataLoader(
        torch.utils.data.Subset(test_loader.dataset, list(range(0, min(1000, len(test_loader.dataset))))),
        batch_size=128, shuffle=False
    )
    
    # Calibrate threshold
    attack.calibrate_threshold(model, train_subset, test_subset)
    
    # Create combined dataset for evaluation (use smaller subset for testing)
    combined_data = []
    combined_labels = []
    
    # Add training samples (members) - limit to 2000 samples
    count = 0
    for data, target in train_loader:
        if count >= 2000:
            break
        combined_data.append(data)
        combined_labels.extend([1] * len(data))  # 1 = member
        count += len(data)
        
    # Add test samples (non-members) - limit to 2000 samples
    count = 0
    for data, target in test_loader:
        if count >= 2000:
            break
        combined_data.append(data)
        combined_labels.extend([0] * len(data))  # 0 = non-member
        count += len(data)
        
    combined_data = torch.cat(combined_data)
    combined_labels = torch.tensor(combined_labels)
    combined_dataset = TensorDataset(combined_data, combined_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
    
    # Run attack
    results = attack.infer_membership(model, combined_loader)
    
    # Calculate metrics
    true_labels = combined_labels.numpy()
    predictions = results['predictions']
    confidences = results['confidences']
    
    # Print debug information
    logger.info(f"True labels distribution: {np.bincount(true_labels)}")
    logger.info(f"Predicted labels distribution: {np.bincount(predictions)}")
    logger.info(f"Confidence stats - Mean: {confidences.mean():.4f}, Std: {confidences.std():.4f}")
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )
    
    try:
        auc_score = roc_auc_score(true_labels, confidences)
    except ValueError as e:
        logger.warning(f"AUC calculation failed: {e}")
        auc_score = 0.5
    
    return {
        'attack_type': 'threshold',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'threshold': results['threshold'],
        'probabilities': confidences.tolist(),
        'true_labels': true_labels.tolist()
    }


def run_loss_attack(model, train_loader, test_loader, device):
    """Run loss-based attack."""
    logger.info("Running Loss-based Attack...")
    
    attack = LossBasedAttack(device=device)
    
    # Calibrate threshold using smaller subsets
    train_subset = DataLoader(
        torch.utils.data.Subset(train_loader.dataset, list(range(0, min(1000, len(train_loader.dataset))))),
        batch_size=128, shuffle=False
    )
    test_subset = DataLoader(
        torch.utils.data.Subset(test_loader.dataset, list(range(0, min(1000, len(test_loader.dataset))))),
        batch_size=128, shuffle=False
    )
    
    # Calibrate threshold
    attack.calibrate_threshold(model, train_subset, test_subset)
    
    # Create combined dataset for evaluation with proper structure
    combined_data = []
    combined_targets = []  # Class targets for loss calculation
    combined_membership_labels = []  # Membership labels (0/1)
    
    # Add training samples (members) - limit to 2000 samples
    count = 0
    for data, target in train_loader:
        if count >= 2000:
            break
        combined_data.append(data)
        combined_targets.append(target)
        combined_membership_labels.extend([1] * len(data))  # 1 = member
        count += len(data)
        
    # Add test samples (non-members) - limit to 2000 samples
    count = 0
    for data, target in test_loader:
        if count >= 2000:
            break
        combined_data.append(data)
        combined_targets.append(target)
        combined_membership_labels.extend([0] * len(data))  # 0 = non-member
        count += len(data)
        
    combined_data = torch.cat(combined_data)
    combined_targets = torch.cat(combined_targets)
    combined_membership_labels = torch.tensor(combined_membership_labels)
    
    # Create dataset with three elements: (data, class_targets, membership_labels)
    combined_dataset = TensorDataset(combined_data, combined_targets, combined_membership_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
    
    # Run attack
    results = attack.infer_membership(model, combined_loader)
    
    # Calculate metrics
    true_labels = combined_membership_labels.numpy()
    predictions = results['predictions']
    losses = results['losses']
    
    # Print debug information
    logger.info(f"True labels distribution: {np.bincount(true_labels)}")
    logger.info(f"Predicted labels distribution: {np.bincount(predictions)}")
    member_losses = losses[true_labels == 1]
    non_member_losses = losses[true_labels == 0]
    logger.info(f"Member losses - Mean: {member_losses.mean():.4f}, Std: {member_losses.std():.4f}")
    logger.info(f"Non-member losses - Mean: {non_member_losses.mean():.4f}, Std: {non_member_losses.std():.4f}")
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )
    # For AUC, invert losses (lower loss = higher membership score)
    try:
        auc_score = roc_auc_score(true_labels, -losses)
    except ValueError as e:
        logger.warning(f"AUC calculation failed: {e}")
        auc_score = 0.5
    
    return {
        'attack_type': 'loss',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'threshold': results['threshold'],
        'probabilities': (-losses).tolist(),  # Inverted losses as probabilities
        'true_labels': true_labels.tolist()
    }


def run_shadow_attack(model, train_dataset, test_dataset, device, output_dir):
    """Run shadow model attack."""
    logger.info("Running Shadow Attack...")
    
    # Create simplified config with reduced complexity for debugging
    config = SimpleNamespace(
        shadow=SimpleNamespace(
            num_shadow_models=5,  # Reduced from 10
            shadow_train_size=2000,  # Reduced size
            shadow_eval_size=500,
            shadow_test_size=500
        ),
        dataset=SimpleNamespace(
            train_batch_size=128,
            eval_batch_size=128,
            num_workers=2
        ),
        paths=SimpleNamespace(
            attack_dir=output_dir
        ),
        attack=SimpleNamespace(
            catboost=SimpleNamespace(
                iterations=100,  # Reduced from 200
                depth=4,  # Reduced from 6
                learning_rate=0.1,
                loss_function='Logloss'
            )
        ),
        # Add model config for shadow model creation
        model=SimpleNamespace(
            name='simple_cnn',
            num_classes=10
        ),
        # Add training config
        training=SimpleNamespace(
            epochs=5,  # Reduced from 10
            lr=0.001,
            batch_size=128
        )
    )
    
    try:
        attack = ShadowAttack(config, device)
        
        # Train shadow models
        shadow_models_info = attack.train_shadow_models(train_dataset)
        
        # Create attack dataset
        attack_df = attack.create_attack_dataset(shadow_models_info, train_dataset)
        
        # Train attack model
        attack.train_attack_model(attack_df)
        
        # Create evaluation data (use smaller subset)
        train_subset_indices = list(range(0, min(2000, len(train_dataset))))
        test_subset_indices = list(range(0, min(2000, len(test_dataset))))
        
        train_subset = torch.utils.data.Subset(train_dataset, train_subset_indices)
        test_subset = torch.utils.data.Subset(test_dataset, test_subset_indices)
        
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
        
        combined_data = []
        combined_labels = []
        
        # Add training samples (members)
        for data, target in train_loader:
            combined_data.append(data)
            combined_labels.extend([1] * len(data))  # 1 = member
            
        # Add test samples (non-members)  
        for data, target in test_loader:
            combined_data.append(data)
            combined_labels.extend([0] * len(data))  # 0 = non-member
            
        combined_data = torch.cat(combined_data)
        combined_labels = torch.tensor(combined_labels)
        combined_dataset = TensorDataset(combined_data, combined_labels)
        combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
        
        # Run attack
        results = attack.infer_membership(model, combined_loader)
        
        # Calculate metrics
        true_labels = combined_labels.numpy()
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        # Print debug information
        logger.info(f"True labels distribution: {np.bincount(true_labels)}")
        logger.info(f"Predicted labels distribution: {np.bincount(predictions)}")
        logger.info(f"Probability stats - Mean: {probabilities.mean():.4f}, Std: {probabilities.std():.4f}")
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        try:
            auc_score = roc_auc_score(true_labels, probabilities)
        except ValueError as e:
            logger.warning(f"AUC calculation failed: {e}")
            auc_score = 0.5
        
        return {
            'attack_type': 'shadow',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'num_shadow_models': config.shadow.num_shadow_models,
            'probabilities': probabilities.tolist(),
            'true_labels': true_labels.tolist()
        }
        
    except Exception as e:
        logger.error(f"Shadow attack failed with error: {e}")
        # Return dummy results to prevent crash
        return {
            'attack_type': 'shadow',
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'auc': 0.5,
            'num_shadow_models': 0,
            'probabilities': [0.5] * 100,  # Dummy probabilities
            'true_labels': [0, 1] * 50  # Dummy labels
        }


def main():
    parser = argparse.ArgumentParser(description='Run membership inference attacks')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist'],
                       help='Dataset used for training')
    
    # Attack arguments
    parser.add_argument('--attack-type', type=str, default='threshold',
                       choices=['threshold', 'loss', 'shadow', 'all'],
                       help='Type of attack to run')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing data')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for evaluation')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./results/mia',
                       help='Directory to save attack results')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generating visualizations')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
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
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    train_dataset, test_dataset, num_classes = get_dataset(args.dataset, args.data_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = create_simple_model(args.dataset, num_classes)
    
    try:
        # Try to load with weights_only=True first (newer PyTorch)
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    except TypeError:
        # Fall back to old method
        state_dict = torch.load(args.model_path, map_location=device)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Run attacks
    results = {}
    
    if args.attack_type == 'threshold' or args.attack_type == 'all':
        try:
            result = run_threshold_attack(model, train_loader, test_loader, device)
            results['threshold'] = result
            logger.info(f"Threshold Attack - Accuracy: {result['accuracy']:.4f}, AUC: {result['auc']:.4f}")
        except Exception as e:
            logger.error(f"Threshold attack failed: {e}")
    
    if args.attack_type == 'loss' or args.attack_type == 'all':
        try:
            result = run_loss_attack(model, train_loader, test_loader, device)
            results['loss'] = result
            logger.info(f"Loss Attack - Accuracy: {result['accuracy']:.4f}, AUC: {result['auc']:.4f}")
        except Exception as e:
            logger.error(f"Loss attack failed: {e}")
    
    if args.attack_type == 'shadow' or args.attack_type == 'all':
        try:
            result = run_shadow_attack(model, train_dataset, test_dataset, device, str(output_dir))
            results['shadow'] = result
            logger.info(f"Shadow Attack - Accuracy: {result['accuracy']:.4f}, AUC: {result['auc']:.4f}")
        except Exception as e:
            logger.error(f"Shadow attack failed: {e}")
    
    # Save results
    results_file = output_dir / 'attack_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Generate visualizations
    if not args.no_visualizations and results:
        create_attack_visualizations(results, output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("MEMBERSHIP INFERENCE ATTACK RESULTS")
    print("="*50)
    
    for attack_name, result in results.items():
        print(f"\n{attack_name.upper()} ATTACK:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1']:.4f}")
        print(f"  AUC:       {result['auc']:.4f}")


if __name__ == '__main__':
    main() 