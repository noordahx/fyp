#!/usr/bin/env python3
"""
Membership Inference Attack Runner

This script runs comprehensive membership inference attacks on trained models
to evaluate privacy risks.

Usage:
    python scripts/mia_runner.py --model results/model.pt --dataset cifar10 --attack shadow
    python scripts/mia_runner.py --model results/model.pt --dataset mnist --attack reference
    python scripts/mia_runner.py --model results/model.pt --dataset cifar10 --attack loo
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


def load_model(model_path, num_classes, device):
    """Load a trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    model = create_model(num_classes, pretrained=False)
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
    """Run simplified shadow model attack."""
    logger.info("Running simplified shadow model attack...")
    
    # Create simple shadow models
    shadow_models = []
    shadow_predictions = []
    shadow_labels = []
    
    # Train a few simple shadow models
    for i in range(args.num_shadows):
        logger.info(f"Training shadow model {i+1}/{args.num_shadows}")
        
        # Create model
        shadow_model = create_model(args.num_classes, pretrained=False)
        shadow_model = shadow_model.to(device)
        
        # Create shadow training data (subset of original)
        shadow_size = min(len(train_dataset) // args.num_shadows, 5000)
        shadow_indices = np.random.choice(len(train_dataset), shadow_size, replace=False)
        shadow_train_data = Subset(train_dataset, shadow_indices)
        shadow_train_loader = DataLoader(shadow_train_data, batch_size=128, shuffle=True)
        
        # Train shadow model (simplified)
        optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(args.shadow_epochs):
            shadow_model.train()
            for batch_x, batch_y in shadow_train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = shadow_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Collect shadow model predictions
        shadow_model.eval()
        with torch.no_grad():
            # Member predictions (training data)
            for batch_x, batch_y in DataLoader(shadow_train_data, batch_size=128):
                batch_x = batch_x.to(device)
                outputs = shadow_model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                shadow_predictions.extend(max_probs.cpu().numpy())
                shadow_labels.extend([1] * len(batch_x))  # Members
            
            # Non-member predictions (random test data)
            test_subset_size = min(len(test_dataset), shadow_size)
            test_indices = np.random.choice(len(test_dataset), test_subset_size, replace=False)
            test_subset = Subset(test_dataset, test_indices)
            for batch_x, batch_y in DataLoader(test_subset, batch_size=128):
                batch_x = batch_x.to(device)
                outputs = shadow_model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                shadow_predictions.extend(max_probs.cpu().numpy())
                shadow_labels.extend([0] * len(batch_x))  # Non-members
    
    # Train simple attack model using confidence scores
    from sklearn.linear_model import LogisticRegression
    
    X = np.array(shadow_predictions).reshape(-1, 1)
    y = np.array(shadow_labels)
    
    attack_model = LogisticRegression()
    attack_model.fit(X, y)
    
    # Evaluate on target model
    member_dataset, non_member_dataset, _, _ = create_attack_splits(
        train_dataset, list(range(len(train_dataset))), 
        list(range(len(test_dataset))), args.attack_size
    )
    
    # Get target model predictions
    target_predictions = []
    target_labels = []
    
    target_model.eval()
    with torch.no_grad():
        # Member predictions
        for batch_x, batch_y in DataLoader(member_dataset, batch_size=128):
            batch_x = batch_x.to(device)
            outputs = target_model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            target_predictions.extend(max_probs.cpu().numpy())
            target_labels.extend([1] * len(batch_x))
        
        # Non-member predictions
        for batch_x, batch_y in DataLoader(non_member_dataset, batch_size=128):
            batch_x = batch_x.to(device)
            outputs = target_model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            target_predictions.extend(max_probs.cpu().numpy())
            target_labels.extend([0] * len(batch_x))
    
    # Attack target model
    X_target = np.array(target_predictions).reshape(-1, 1)
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
                       choices=['shadow', 'threshold', 'all'],
                       help='Type of attack to run')
    parser.add_argument('--attack-size', type=int, default=1000,
                       help='Number of samples for attack evaluation')
    
    # Shadow attack arguments
    parser.add_argument('--num-shadows', type=int, default=3,
                       help='Number of shadow models')
    parser.add_argument('--shadow-epochs', type=int, default=5,
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
    target_model = load_model(args.model, num_classes, device)
    
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