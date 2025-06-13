#!/usr/bin/env python3
"""
Membership Inference Attack Runner

This script runs comprehensive membership inference attacks on trained models
to evaluate privacy risks.

Usage:
    python scripts/mia_runner.py --model results/model.pt --dataset cifar10 --attack shadow
    python scripts/mia_runner.py --model results/model.pt --dataset mnist --attack reference
    python scripts/mia_runner.py --model results/model.pt --dataset cifar10 --attack loo
    python scripts/mia_runner.py --model results/model.pt.0.pt --dataset mnist --attack all --attack-size 500 --num-shadows 2 --shadow-epochs 5
"""

import argparse
import logging
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mia_lib.data import get_dataset
from mia_lib.models import create_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_model(model_path, num_classes, device, dataset_name):
    """Load a trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    # Use same model architecture logic as training
    input_channels = 1 if dataset_name == 'mnist' else 3
    model = create_model(num_classes, pretrained=False, input_channels=input_channels)
    model = model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("Model loaded successfully")
    return model


def create_attack_splits(dataset, train_indices, test_indices, attack_size=1000):
    """Create member and non-member datasets for attack."""
    # Take subset of training data as members
    if len(train_indices) > attack_size:
        member_indices = np.random.choice(train_indices, attack_size, replace=False)
    else:
        member_indices = train_indices
    
    # Take subset of test data as non-members
    if len(test_indices) > attack_size:
        non_member_indices = np.random.choice(test_indices, attack_size, replace=False)
    else:
        non_member_indices = test_indices
    
    member_dataset = Subset(dataset, member_indices)
    non_member_dataset = Subset(dataset, non_member_indices)
    
    return member_dataset, non_member_dataset, member_indices, non_member_indices


def evaluate_attack_performance(predictions, labels, save_path=None):
    """Evaluate and visualize attack performance."""
    # Calculate metrics
    auc = roc_auc_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions > 0.5)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(labels, predictions)
    
    logger.info(f"Attack AUC: {auc:.4f}")
    logger.info(f"Attack Accuracy: {accuracy:.4f}")
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot ROC curve and precision-recall curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, predictions)
        ax1.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True)
        
        # Precision-Recall curve
        ax2.plot(recall, precision, label=f'PR Curve')
        ax2.axhline(y=0.5, color='k', linestyle='--', label='Random')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / 'attack_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attack performance plots saved to {save_path}")
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }


def run_shadow_attack_simple(target_model, train_dataset, test_dataset, device, output_dir, args):
    """Run enhanced shadow model attack with multiple features."""
    logger.info("Running enhanced shadow model attack...")
    
    # Collect shadow training data
    shadow_features = []
    shadow_labels = []
    
    # Train shadow models
    for i in range(args.num_shadows):
        logger.info(f"Training shadow model {i+1}/{args.num_shadows}")
        
        # Create model with same architecture as target
        input_channels = 1 if 'mnist' in str(args.dataset).lower() else 3
        shadow_model = create_model(args.num_classes, pretrained=False, input_channels=input_channels)
        shadow_model = shadow_model.to(device)
        
        # Create disjoint shadow training data
        total_size = len(train_dataset) + len(test_dataset)
        shadow_size = min(total_size // (args.num_shadows * 2), 3000)
        
        # Create shadow train/test splits
        all_indices = list(range(len(train_dataset))) + [len(train_dataset) + i for i in range(len(test_dataset))]
        shadow_train_indices = np.random.choice(all_indices, shadow_size, replace=False)
        remaining_indices = [idx for idx in all_indices if idx not in shadow_train_indices]
        shadow_test_indices = np.random.choice(remaining_indices, shadow_size, replace=False)
        
        # Create shadow datasets
        shadow_train_samples = []
        shadow_test_samples = []
        
        for idx in shadow_train_indices:
            if idx < len(train_dataset):
                shadow_train_samples.append(train_dataset[idx])
            else:
                shadow_test_samples.append(test_dataset[idx - len(train_dataset)])
        
        for idx in shadow_test_indices:
            if idx < len(train_dataset):
                shadow_test_samples.append(train_dataset[idx])
            else:
                shadow_test_samples.append(test_dataset[idx - len(train_dataset)])
        
        # Convert to proper datasets
        shadow_train_loader = DataLoader(shadow_train_samples, batch_size=64, shuffle=True)
        shadow_test_loader = DataLoader(shadow_test_samples, batch_size=64, shuffle=False)
        
        # Train shadow model with proper training loop
        optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
        
        for epoch in range(args.shadow_epochs):
            shadow_model.train()
            for batch_x, batch_y in shadow_train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = shadow_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        # Extract features from shadow model
        shadow_model.eval()
        criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            # Member features (shadow training data)
            for batch_x, batch_y in shadow_train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = shadow_model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                losses = criterion_no_reduction(outputs, batch_y)
                
                # Extract multiple features
                for j in range(len(batch_x)):
                    features = [
                        float(torch.max(probs[j])),  # Max probability
                        float(torch.sum(probs[j] ** 2)),  # Sum of squared probabilities  
                        float(losses[j]),  # Cross-entropy loss
                        float(torch.std(probs[j])),  # Standard deviation of probabilities
                        float(probs[j][batch_y[j]]),  # Probability of true class
                    ]
                    shadow_features.append(features)
                    shadow_labels.append(1)  # Member
            
            # Non-member features (shadow test data)
            for batch_x, batch_y in shadow_test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = shadow_model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                losses = criterion_no_reduction(outputs, batch_y)
                
                # Extract multiple features
                for j in range(len(batch_x)):
                    features = [
                        float(torch.max(probs[j])),  # Max probability
                        float(torch.sum(probs[j] ** 2)),  # Sum of squared probabilities  
                        float(losses[j]),  # Cross-entropy loss
                        float(torch.std(probs[j])),  # Standard deviation of probabilities
                        float(probs[j][batch_y[j]]),  # Probability of true class
                    ]
                    shadow_features.append(features)
                    shadow_labels.append(0)  # Non-member
    
    # Train sophisticated attack model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    X = np.array(shadow_features)
    y = np.array(shadow_labels)
    
    # Split into train/test for attack model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Use Random Forest for better performance
    attack_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    attack_model.fit(X_train, y_train)
    
    logger.info(f"Shadow attack model trained on {len(X_train)} samples")
    logger.info(f"Shadow attack model validation accuracy: {attack_model.score(X_test, y_test):.4f}")
    
    # Evaluate on target model
    member_dataset, non_member_dataset, member_indices, non_member_indices = create_attack_splits(
        train_dataset, list(range(len(train_dataset))), 
        list(range(len(test_dataset))), args.attack_size
    )
    
    # Extract features from target model
    target_features = []
    target_labels = []
    
    target_model.eval()
    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        # Member features
        member_loader = DataLoader(member_dataset, batch_size=128)
        for i, (batch_x, batch_y) in enumerate(member_loader):
            batch_x = batch_x.to(device)
            # Get true labels
            start_idx = i * 128
            end_idx = min(start_idx + 128, len(member_indices))
            true_labels = []
            for idx in range(start_idx, end_idx):
                if idx < len(member_indices):
                    orig_idx = member_indices[idx]
                    _, true_label = train_dataset[orig_idx]
                    true_labels.append(true_label)
            
            if true_labels:
                true_labels = torch.tensor(true_labels, device=device)
                outputs = target_model(batch_x[:len(true_labels)])
                probs = torch.softmax(outputs, dim=1)
                losses = criterion_no_reduction(outputs, true_labels)
                
                for j in range(len(true_labels)):
                    features = [
                        float(torch.max(probs[j])),
                        float(torch.sum(probs[j] ** 2)),
                        float(losses[j]),
                        float(torch.std(probs[j])),
                        float(probs[j][true_labels[j]]),
                    ]
                    target_features.append(features)
                    target_labels.append(1)
        
        # Non-member features
        non_member_loader = DataLoader(non_member_dataset, batch_size=128)
        for i, (batch_x, batch_y) in enumerate(non_member_loader):
            batch_x = batch_x.to(device)
            # Get true labels
            start_idx = i * 128
            end_idx = min(start_idx + 128, len(non_member_indices))
            true_labels = []
            for idx in range(start_idx, end_idx):
                if idx < len(non_member_indices):
                    orig_idx = non_member_indices[idx]
                    _, true_label = test_dataset[orig_idx]
                    true_labels.append(true_label)
            
            if true_labels:
                true_labels = torch.tensor(true_labels, device=device)
                outputs = target_model(batch_x[:len(true_labels)])
                probs = torch.softmax(outputs, dim=1)
                losses = criterion_no_reduction(outputs, true_labels)
                
                for j in range(len(true_labels)):
                    features = [
                        float(torch.max(probs[j])),
                        float(torch.sum(probs[j] ** 2)),
                        float(losses[j]),
                        float(torch.std(probs[j])),
                        float(probs[j][true_labels[j]]),
                    ]
                    target_features.append(features)
                    target_labels.append(0)
    
    # Attack target model
    X_target = np.array(target_features)
    attack_probs = attack_model.predict_proba(X_target)[:, 1]
    
    # Evaluate performance
    results = evaluate_attack_performance(
        attack_probs, np.array(target_labels), 
        output_dir / 'shadow_attack'
    )
    
    return results


def run_threshold_attack(target_model, train_dataset, test_dataset, device, output_dir, args):
    """Run threshold-based attack."""
    logger.info("Running threshold-based attack...")
    
    member_dataset, non_member_dataset, _, _ = create_attack_splits(
        train_dataset, list(range(len(train_dataset))), 
        list(range(len(test_dataset))), args.attack_size
    )
    
    predictions = []
    labels = []
    
    target_model.eval()
    with torch.no_grad():
        # Member predictions
        for batch_x, batch_y in DataLoader(member_dataset, batch_size=128):
            batch_x = batch_x.to(device)
            outputs = target_model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            predictions.extend(max_probs.cpu().numpy())
            labels.extend([1] * len(batch_x))
        
        # Non-member predictions
        for batch_x, batch_y in DataLoader(non_member_dataset, batch_size=128):
            batch_x = batch_x.to(device)
            outputs = target_model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            predictions.extend(max_probs.cpu().numpy())
            labels.extend([0] * len(batch_x))
    
    # Use confidence scores as attack predictions
    results = evaluate_attack_performance(
        np.array(predictions), np.array(labels),
        output_dir / 'threshold_attack'
    )
    
    return results


def run_loss_based_attack(target_model, train_dataset, test_dataset, device, output_dir, args):
    """Run loss-based membership inference attack."""
    logger.info("Running loss-based attack...")
    
    member_dataset, non_member_dataset, member_indices, non_member_indices = create_attack_splits(
        train_dataset, list(range(len(train_dataset))), 
        list(range(len(test_dataset))), args.attack_size
    )
    
    predictions = []
    labels = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    target_model.eval()
    
    with torch.no_grad():
        # Member losses (training data)
        member_loader = DataLoader(member_dataset, batch_size=128)
        for i, (batch_x, batch_y) in enumerate(member_loader):
            batch_x = batch_x.to(device)
            # Get corresponding true labels from original training data
            start_idx = i * 128
            end_idx = min(start_idx + 128, len(member_indices))
            true_labels = []
            for idx in range(start_idx, end_idx):
                if idx < len(member_indices):
                    orig_idx = member_indices[idx]
                    _, true_label = train_dataset[orig_idx]
                    true_labels.append(true_label)
            
            if true_labels:
                true_labels = torch.tensor(true_labels, device=device)
                outputs = target_model(batch_x[:len(true_labels)])
                losses = criterion(outputs, true_labels)
                # Lower loss = higher membership probability
                # Convert to membership probability (invert loss)
                max_loss = 10.0  # Reasonable upper bound for cross-entropy loss
                membership_probs = 1.0 - torch.clamp(losses / max_loss, 0.0, 1.0)
                predictions.extend(membership_probs.cpu().numpy())
                labels.extend([1] * len(true_labels))
        
        # Non-member losses (test data)
        non_member_loader = DataLoader(non_member_dataset, batch_size=128)
        for i, (batch_x, batch_y) in enumerate(non_member_loader):
            batch_x = batch_x.to(device)
            # Get corresponding true labels from original test data
            start_idx = i * 128
            end_idx = min(start_idx + 128, len(non_member_indices))
            true_labels = []
            for idx in range(start_idx, end_idx):
                if idx < len(non_member_indices):
                    orig_idx = non_member_indices[idx]
                    _, true_label = test_dataset[orig_idx]
                    true_labels.append(true_label)
            
            if true_labels:
                true_labels = torch.tensor(true_labels, device=device)
                outputs = target_model(batch_x[:len(true_labels)])
                losses = criterion(outputs, true_labels)
                # Lower loss = higher membership probability
                max_loss = 10.0  # Reasonable upper bound for cross-entropy loss
                membership_probs = 1.0 - torch.clamp(losses / max_loss, 0.0, 1.0)
                predictions.extend(membership_probs.cpu().numpy())
                labels.extend([0] * len(true_labels))
    
    # Evaluate performance
    results = evaluate_attack_performance(
        np.array(predictions), np.array(labels),
        output_dir / 'loss_attack'
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run membership inference attacks')
    
    # Model and data arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist', 'cifar100'],
                       help='Dataset used to train the model')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing dataset')
    
    # Attack arguments
    parser.add_argument('--attack', type=str, default='shadow',
                       choices=['shadow', 'threshold', 'loss', 'all'],
                       help='Type of attack to run')
    parser.add_argument('--attack-size', type=int, default=1000,
                       help='Number of samples for attack evaluation')
    
    # Shadow attack arguments
    parser.add_argument('--num-shadows', type=int, default=3,
                       help='Number of shadow models')
    parser.add_argument('--shadow-epochs', type=int, default=8,
                       help='Epochs for shadow model training')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./attack_results',
                       help='Directory to save attack results')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    train_dataset, test_dataset, num_classes = get_dataset(args.dataset, args.data_dir)
    args.num_classes = num_classes
    
    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Load target model
    target_model = load_model(args.model, num_classes, device, args.dataset)
    
    # Run attacks
    all_results = {}
    
    if args.attack == 'shadow' or args.attack == 'all':
        shadow_results = run_shadow_attack_simple(
            target_model, train_dataset, test_dataset, device, output_dir, args
        )
        all_results['shadow'] = shadow_results
    
    if args.attack == 'threshold' or args.attack == 'all':
        threshold_results = run_threshold_attack(
            target_model, train_dataset, test_dataset, device, output_dir, args
        )
        all_results['threshold'] = threshold_results
    
    if args.attack == 'loss' or args.attack == 'all':
        loss_results = run_loss_based_attack(
            target_model, train_dataset, test_dataset, device, output_dir, args
        )
        all_results['loss'] = loss_results
    
    # Save results
    results_file = output_dir / 'attack_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Attack results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("MEMBERSHIP INFERENCE ATTACK RESULTS")
    print("="*50)
    
    for attack_name, results in all_results.items():
        print(f"\n{attack_name.upper()} ATTACK:")
        print(f"  AUC: {results['auc']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        
        # Risk assessment
        if results['auc'] > 0.8:
            risk = "HIGH"
        elif results['auc'] > 0.6:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        print(f"  Privacy Risk: {risk}")
    
    print("\n" + "="*50)


if __name__ == '__main__':
    main() 