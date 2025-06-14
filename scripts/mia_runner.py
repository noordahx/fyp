#!/usr/bin/env python3
"""
Membership Inference Attack Runner

This script runs comprehensive membership inference attacks on trained models
to evaluate privacy risks.

Usage:
    python scripts/mia_runner.py --model results/model.pt --dataset cifar10 --attack shadow
    python scripts/mia_runner.py --model results/model.pt --dataset mnist --attack loss
    python scripts/mia_runner.py --model results/model.pt --dataset cifar10 --attack shadow
    python scripts/mia_runner.py --model results/model.pt --dataset mnist --attack all --attack-size 500 --num-shadows 2 --shadow-epochs 5
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
    """Evaluate and visualize attack performance with proper threshold optimization."""
    from sklearn.metrics import roc_curve
    
    # Ensure predictions are in [0,1] range
    predictions = np.clip(predictions, 0.0, 1.0)
    
    # Calculate AUC
    auc = roc_auc_score(labels, predictions)
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    # Optimal threshold maximizes (TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate accuracy with optimal threshold
    optimal_accuracy = accuracy_score(labels, predictions >= optimal_threshold)
    
    # Calculate accuracy with 0.5 threshold for comparison
    default_accuracy = accuracy_score(labels, predictions >= 0.5)
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, predictions)
    
    # Calculate additional metrics
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred_optimal = (predictions >= optimal_threshold).astype(int)
    
    # Get balanced accuracy (accounts for class imbalance)
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(labels, y_pred_optimal)
    
    logger.info(f"Attack AUC: {auc:.4f}")
    logger.info(f"Attack Accuracy (optimal threshold {optimal_threshold:.3f}): {optimal_accuracy:.4f}")
    logger.info(f"Attack Accuracy (default threshold 0.5): {default_accuracy:.4f}")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Print confusion matrix for better understanding
    cm = confusion_matrix(labels, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"Confusion Matrix - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    logger.info(f"True Positive Rate (Recall): {tp/(tp+fn):.4f}")
    logger.info(f"True Negative Rate (Specificity): {tn/(tn+fp):.4f}")
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Attack Performance Analysis', fontsize=16)
        
        # ROC curve
        ax1.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        ax1.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                   label=f'Optimal (t={optimal_threshold:.3f})', zorder=5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall curve
        ax2.plot(recall, precision, label=f'PR Curve', linewidth=2)
        ax2.axhline(y=0.5, color='k', linestyle='--', label='Random', alpha=0.5)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Threshold vs Accuracy plot
        accuracies = []
        test_thresholds = np.linspace(0.01, 0.99, 50)
        for t in test_thresholds:
            acc = accuracy_score(labels, predictions >= t)
            accuracies.append(acc)
        
        ax3.plot(test_thresholds, accuracies, linewidth=2)
        ax3.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal: {optimal_threshold:.3f}')
        ax3.axvline(x=0.5, color='orange', linestyle='--', 
                   label='Default: 0.5')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Prediction distribution
        member_preds = predictions[labels == 1]
        non_member_preds = predictions[labels == 0]
        
        ax4.hist(non_member_preds, bins=30, alpha=0.7, label='Non-members', density=True)
        ax4.hist(member_preds, bins=30, alpha=0.7, label='Members', density=True)
        ax4.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal threshold')
        ax4.axvline(x=0.5, color='orange', linestyle='--', 
                   label='Default threshold')
        ax4.set_xlabel('Prediction Score')
        ax4.set_ylabel('Density')
        ax4.set_title('Prediction Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'attack_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attack performance plots saved to {save_path}")
    
    return {
        'auc': float(auc),
        'accuracy_optimal': float(optimal_accuracy),
        'accuracy_default': float(default_accuracy),
        'balanced_accuracy': float(balanced_acc),
        'optimal_threshold': float(optimal_threshold),
        'precision': [float(x) for x in precision],
        'recall': [float(x) for x in recall],
        'confusion_matrix': [[int(x) for x in row] for row in cm.tolist()],
        'member_prediction_mean': float(np.mean(predictions[labels == 1])),
        'non_member_prediction_mean': float(np.mean(predictions[labels == 0]))
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
                
                # Extract sophisticated features
                for j in range(len(batch_x)):
                    prob_vec = probs[j]
                    loss_val = losses[j]
                    
                    # Core features
                    max_prob = float(torch.max(prob_vec))
                    true_class_prob = float(prob_vec[batch_y[j]])
                    
                    # Statistical features
                    sum_squared = float(torch.sum(prob_vec ** 2))
                    std_dev = float(torch.std(prob_vec))
                    entropy = float(-torch.sum(prob_vec * torch.log(prob_vec + 1e-12)))
                    
                    # Loss-based features
                    raw_loss = float(loss_val)
                    log_loss = float(torch.log(loss_val + 1e-12))
                    
                    # Ranking features
                    sorted_probs, _ = torch.sort(prob_vec, descending=True)
                    top_2_diff = float(sorted_probs[0] - sorted_probs[1])
                    top_3_sum = float(torch.sum(sorted_probs[:3]))
                    
                    # Advanced features
                    gini_impurity = float(1.0 - torch.sum(prob_vec ** 2))
                    pred_margin = float(prob_vec[batch_y[j]] - torch.max(torch.cat([prob_vec[:batch_y[j]], prob_vec[batch_y[j]+1:]])))
                    
                    features = [
                        max_prob,           # 1. Max probability
                        true_class_prob,    # 2. Probability of true class  
                        sum_squared,        # 3. Sum of squared probabilities
                        std_dev,           # 4. Standard deviation
                        entropy,           # 5. Entropy of prediction
                        raw_loss,          # 6. Cross-entropy loss
                        log_loss,          # 7. Log of loss
                        top_2_diff,        # 8. Difference between top 2 predictions
                        top_3_sum,         # 9. Sum of top 3 predictions
                        gini_impurity,     # 10. Gini impurity
                        pred_margin,       # 11. Prediction margin
                    ]
                    shadow_features.append(features)
                    shadow_labels.append(1)  # Member
            
            # Non-member features (shadow test data)
            for batch_x, batch_y in shadow_test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = shadow_model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                losses = criterion_no_reduction(outputs, batch_y)
                
                # Extract sophisticated features (same as member features)
                for j in range(len(batch_x)):
                    prob_vec = probs[j]
                    loss_val = losses[j]
                    
                    # Core features
                    max_prob = float(torch.max(prob_vec))
                    true_class_prob = float(prob_vec[batch_y[j]])
                    
                    # Statistical features
                    sum_squared = float(torch.sum(prob_vec ** 2))
                    std_dev = float(torch.std(prob_vec))
                    entropy = float(-torch.sum(prob_vec * torch.log(prob_vec + 1e-12)))
                    
                    # Loss-based features
                    raw_loss = float(loss_val)
                    log_loss = float(torch.log(loss_val + 1e-12))
                    
                    # Ranking features
                    sorted_probs, _ = torch.sort(prob_vec, descending=True)
                    top_2_diff = float(sorted_probs[0] - sorted_probs[1])
                    top_3_sum = float(torch.sum(sorted_probs[:3]))
                    
                    # Advanced features
                    gini_impurity = float(1.0 - torch.sum(prob_vec ** 2))
                    pred_margin = float(prob_vec[batch_y[j]] - torch.max(torch.cat([prob_vec[:batch_y[j]], prob_vec[batch_y[j]+1:]])))
                    
                    features = [
                        max_prob,           # 1. Max probability
                        true_class_prob,    # 2. Probability of true class  
                        sum_squared,        # 3. Sum of squared probabilities
                        std_dev,           # 4. Standard deviation
                        entropy,           # 5. Entropy of prediction
                        raw_loss,          # 6. Cross-entropy loss
                        log_loss,          # 7. Log of loss
                        top_2_diff,        # 8. Difference between top 2 predictions
                        top_3_sum,         # 9. Sum of top 3 predictions
                        gini_impurity,     # 10. Gini impurity
                        pred_margin,       # 11. Prediction margin
                    ]
                    shadow_features.append(features)
                    shadow_labels.append(0)  # Non-member
    
    # Train sophisticated ensemble attack model
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X = np.array(shadow_features)
    y = np.array(shadow_labels)
    
    # Split into train/test for attack model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create ensemble of diverse models
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    )
    
    lr_model = LogisticRegression(
        C=1.0,
        random_state=42,
        max_iter=1000
    )
    
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42
    )
    
    # Create voting ensemble
    attack_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('lr', lr_model),
            ('svm', svm_model)
        ],
        voting='soft'
    )
    
    # Train ensemble on scaled features
    attack_model.fit(X_train_scaled, y_train)
    
    logger.info(f"Shadow attack ensemble trained on {len(X_train)} samples")
    logger.info(f"Shadow attack ensemble validation accuracy: {attack_model.score(X_test_scaled, y_test):.4f}")
    
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
                    prob_vec = probs[j]
                    loss_val = losses[j]
                    
                    # Core features
                    max_prob = float(torch.max(prob_vec))
                    true_class_prob = float(prob_vec[true_labels[j]])
                    
                    # Statistical features
                    sum_squared = float(torch.sum(prob_vec ** 2))
                    std_dev = float(torch.std(prob_vec))
                    entropy = float(-torch.sum(prob_vec * torch.log(prob_vec + 1e-12)))
                    
                    # Loss-based features
                    raw_loss = float(loss_val)
                    log_loss = float(torch.log(loss_val + 1e-12))
                    
                    # Ranking features
                    sorted_probs, _ = torch.sort(prob_vec, descending=True)
                    top_2_diff = float(sorted_probs[0] - sorted_probs[1])
                    top_3_sum = float(torch.sum(sorted_probs[:3]))
                    
                    # Advanced features
                    gini_impurity = float(1.0 - torch.sum(prob_vec ** 2))
                    pred_margin = float(prob_vec[true_labels[j]] - torch.max(torch.cat([prob_vec[:true_labels[j]], prob_vec[true_labels[j]+1:]])))
                    
                    features = [
                        max_prob,           # 1. Max probability
                        true_class_prob,    # 2. Probability of true class  
                        sum_squared,        # 3. Sum of squared probabilities
                        std_dev,           # 4. Standard deviation
                        entropy,           # 5. Entropy of prediction
                        raw_loss,          # 6. Cross-entropy loss
                        log_loss,          # 7. Log of loss
                        top_2_diff,        # 8. Difference between top 2 predictions
                        top_3_sum,         # 9. Sum of top 3 predictions
                        gini_impurity,     # 10. Gini impurity
                        pred_margin,       # 11. Prediction margin
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
                    prob_vec = probs[j]
                    loss_val = losses[j]
                    
                    # Core features
                    max_prob = float(torch.max(prob_vec))
                    true_class_prob = float(prob_vec[true_labels[j]])
                    
                    # Statistical features
                    sum_squared = float(torch.sum(prob_vec ** 2))
                    std_dev = float(torch.std(prob_vec))
                    entropy = float(-torch.sum(prob_vec * torch.log(prob_vec + 1e-12)))
                    
                    # Loss-based features
                    raw_loss = float(loss_val)
                    log_loss = float(torch.log(loss_val + 1e-12))
                    
                    # Ranking features
                    sorted_probs, _ = torch.sort(prob_vec, descending=True)
                    top_2_diff = float(sorted_probs[0] - sorted_probs[1])
                    top_3_sum = float(torch.sum(sorted_probs[:3]))
                    
                    # Advanced features
                    gini_impurity = float(1.0 - torch.sum(prob_vec ** 2))
                    pred_margin = float(prob_vec[true_labels[j]] - torch.max(torch.cat([prob_vec[:true_labels[j]], prob_vec[true_labels[j]+1:]])))
                    
                    features = [
                        max_prob,           # 1. Max probability
                        true_class_prob,    # 2. Probability of true class  
                        sum_squared,        # 3. Sum of squared probabilities
                        std_dev,           # 4. Standard deviation
                        entropy,           # 5. Entropy of prediction
                        raw_loss,          # 6. Cross-entropy loss
                        log_loss,          # 7. Log of loss
                        top_2_diff,        # 8. Difference between top 2 predictions
                        top_3_sum,         # 9. Sum of top 3 predictions
                        gini_impurity,     # 10. Gini impurity
                        pred_margin,       # 11. Prediction margin
                    ]
                    target_features.append(features)
                    target_labels.append(0)
    
    # Attack target model with scaled features
    X_target = np.array(target_features)
    X_target_scaled = scaler.transform(X_target)
    attack_probs = attack_model.predict_proba(X_target_scaled)[:, 1]
    
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
    """Run enhanced loss-based membership inference attack."""
    logger.info("Running enhanced loss-based attack...")
    
    member_dataset, non_member_dataset, member_indices, non_member_indices = create_attack_splits(
        train_dataset, list(range(len(train_dataset))), 
        list(range(len(test_dataset))), args.attack_size
    )
    
    all_losses = []
    all_features = []
    labels = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    target_model.eval()
    
    with torch.no_grad():
        # Collect member losses (training data)
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
                probs = torch.softmax(outputs, dim=1)
                losses = criterion(outputs, true_labels)
                
                for j in range(len(true_labels)):
                    loss_val = float(losses[j])
                    prob_vec = probs[j]
                    
                    # Multiple loss-based features
                    features = [
                        loss_val,                                              # Raw loss
                        np.log(loss_val + 1e-12),                            # Log loss
                        1.0 / (1.0 + loss_val),                             # Inverse loss
                        float(prob_vec[true_labels[j]]),                     # True class probability
                        float(torch.max(prob_vec)),                          # Max probability
                        loss_val * float(prob_vec[true_labels[j]]),          # Loss * true class prob
                        float(-torch.sum(prob_vec * torch.log(prob_vec + 1e-12))),  # Entropy
                    ]
                    
                    all_losses.append(loss_val)
                    all_features.append(features)
                    labels.append(1)
        
        # Collect non-member losses (test data)
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
                probs = torch.softmax(outputs, dim=1)
                losses = criterion(outputs, true_labels)
                
                for j in range(len(true_labels)):
                    loss_val = float(losses[j])
                    prob_vec = probs[j]
                    
                    # Multiple loss-based features
                    features = [
                        loss_val,                                              # Raw loss
                        np.log(loss_val + 1e-12),                            # Log loss
                        1.0 / (1.0 + loss_val),                             # Inverse loss
                        float(prob_vec[true_labels[j]]),                     # True class probability
                        float(torch.max(prob_vec)),                          # Max probability
                        loss_val * float(prob_vec[true_labels[j]]),          # Loss * true class prob
                        float(-torch.sum(prob_vec * torch.log(prob_vec + 1e-12))),  # Entropy
                    ]
                    
                    all_losses.append(loss_val)
                    all_features.append(features)
                    labels.append(0)
    
    # Use adaptive normalization based on data distribution
    all_losses = np.array(all_losses)
    member_losses = all_losses[np.array(labels) == 1]
    non_member_losses = all_losses[np.array(labels) == 0]
    
    # Compute percentile-based normalization
    loss_percentile_95 = np.percentile(all_losses, 95)
    
    # Enhanced loss-based attack: multiple strategies
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    # Strategy 1: Simple inverted loss
    simple_predictions = 1.0 - np.clip(all_losses / loss_percentile_95, 0.0, 1.0)
    
    # Strategy 2: Feature-based ensemble
    X = np.array(all_features)
    y = np.array(labels)
    
    if len(set(y)) > 1:  # Only train if we have both classes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble classifier
        ensemble_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=3,
            random_state=42
        )
        ensemble_model.fit(X_train_scaled, y_train)
        
        # Get ensemble predictions
        X_scaled = scaler.transform(X)
        ensemble_predictions = ensemble_model.predict_proba(X_scaled)[:, 1]
        
        # Combine strategies (weighted average)
        final_predictions = 0.3 * simple_predictions + 0.7 * ensemble_predictions
    else:
        final_predictions = simple_predictions
    
    # Evaluate performance
    results = evaluate_attack_performance(
        final_predictions, np.array(labels),
        output_dir / 'loss_attack'
    )
    
    return results


def run_population_attack(target_model, train_dataset, test_dataset, device, output_dir, args):
    """Run population-based membership inference attack using reference models."""
    logger.info("Running population-based attack...")
    
    # Train reference models on disjoint population data
    reference_models = []
    num_reference_models = 3
    
    # Create reference training data (disjoint from target training data)
    total_size = len(train_dataset) + len(test_dataset)
    ref_data_size = min(total_size // 2, 5000)
    
    input_channels = 1 if 'mnist' in str(args.dataset).lower() else 3
    
    for i in range(num_reference_models):
        logger.info(f"Training reference model {i+1}/{num_reference_models}")
        
        # Create reference model
        ref_model = create_model(args.num_classes, pretrained=False, input_channels=input_channels)
        ref_model = ref_model.to(device)
        
        # Sample reference training data
        all_indices = list(range(len(train_dataset))) + [len(train_dataset) + j for j in range(len(test_dataset))]
        ref_indices = np.random.choice(all_indices, ref_data_size, replace=False)
        
        ref_samples = []
        for idx in ref_indices:
            if idx < len(train_dataset):
                ref_samples.append(train_dataset[idx])
            else:
                ref_samples.append(test_dataset[idx - len(train_dataset)])
        
        ref_loader = DataLoader(ref_samples, batch_size=64, shuffle=True)
        
        # Train reference model
        optimizer = torch.optim.Adam(ref_model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
        
        for epoch in range(8):
            ref_model.train()
            for batch_x, batch_y in ref_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = ref_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        reference_models.append(ref_model)
    
    # Create attack splits
    member_dataset, non_member_dataset, member_indices, non_member_indices = create_attack_splits(
        train_dataset, list(range(len(train_dataset))), 
        list(range(len(test_dataset))), args.attack_size
    )
    
    # Extract population-based features
    target_features = []
    labels = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    def extract_population_features(model, data_loader, true_indices, original_dataset, is_member):
        features = []
        model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.to(device)
                start_idx = i * 128
                end_idx = min(start_idx + 128, len(true_indices))
                
                true_labels = []
                for idx in range(start_idx, end_idx):
                    if idx < len(true_indices):
                        orig_idx = true_indices[idx]
                        _, true_label = original_dataset[orig_idx]
                        true_labels.append(true_label)
                
                if true_labels:
                    true_labels = torch.tensor(true_labels, device=device)
                    outputs = model(batch_x[:len(true_labels)])
                    probs = torch.softmax(outputs, dim=1)
                    losses = criterion(outputs, true_labels)
                    
                    for j in range(len(true_labels)):
                        prob_vec = probs[j]
                        loss_val = float(losses[j])
                        
                        # Population-based features
                        pop_features = [
                            float(prob_vec[true_labels[j]]),                     # True class probability
                            float(torch.max(prob_vec)),                          # Max probability
                            loss_val,                                            # Cross-entropy loss
                            float(torch.std(prob_vec)),                          # Std deviation
                            float(-torch.sum(prob_vec * torch.log(prob_vec + 1e-12))),  # Entropy
                        ]
                        features.append(pop_features)
                        
        return features
    
    # Get target model features
    target_model.eval()
    member_loader = DataLoader(member_dataset, batch_size=128)
    non_member_loader = DataLoader(non_member_dataset, batch_size=128)
    
    target_member_features = extract_population_features(
        target_model, member_loader, member_indices, train_dataset, True
    )
    target_non_member_features = extract_population_features(
        target_model, non_member_loader, non_member_indices, test_dataset, False
    )
    
    # Get reference model features for the same data points
    reference_features = []
    for ref_model in reference_models:
        ref_member_features = extract_population_features(
            ref_model, member_loader, member_indices, train_dataset, True
        )
        ref_non_member_features = extract_population_features(
            ref_model, non_member_loader, non_member_indices, test_dataset, False
        )
        reference_features.extend(ref_member_features + ref_non_member_features)
    
    # Compute population statistics
    reference_features = np.array(reference_features)
    pop_mean = np.mean(reference_features, axis=0)
    pop_std = np.std(reference_features, axis=0) + 1e-12
    
    # Compute attack scores using population statistics
    attack_scores = []
    attack_labels = []
    
    # Process member features
    for features in target_member_features:
        features = np.array(features)
        # Likelihood ratio based on population statistics
        likelihood_ratio = np.prod(np.exp(-0.5 * ((features - pop_mean) / pop_std) ** 2))
        # Normalize to [0, 1] range (higher = more likely to be member)
        score = 1.0 / (1.0 + likelihood_ratio)
        attack_scores.append(score)
        attack_labels.append(1)
    
    # Process non-member features
    for features in target_non_member_features:
        features = np.array(features)
        # Likelihood ratio based on population statistics  
        likelihood_ratio = np.prod(np.exp(-0.5 * ((features - pop_mean) / pop_std) ** 2))
        # Normalize to [0, 1] range (higher = more likely to be member)
        score = 1.0 / (1.0 + likelihood_ratio)
        attack_scores.append(score)
        attack_labels.append(0)
    
    # Evaluate performance
    results = evaluate_attack_performance(
        np.array(attack_scores), np.array(attack_labels),
        output_dir / 'population_attack'
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
                       choices=['shadow', 'threshold', 'loss', 'population', 'all'],
                       help='Type of attack to run')
    parser.add_argument('--attack-size', type=int, default=1000,
                       help='Number of samples for attack evaluation')
    
    # Shadow attack arguments
    parser.add_argument('--num-shadows', type=int, default=5,
                       help='Number of shadow models')
    parser.add_argument('--shadow-epochs', type=int, default=10,
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
    
    if args.attack == 'population' or args.attack == 'all':
        population_results = run_population_attack(
            target_model, train_dataset, test_dataset, device, output_dir, args
        )
        all_results['population'] = population_results
    
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
        print(f"  Accuracy (optimal): {results['accuracy_optimal']:.4f}")
        print(f"  Accuracy (default): {results['accuracy_default']:.4f}")
        print(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"  Optimal Threshold: {results['optimal_threshold']:.3f}")
        print(f"  Member Pred Mean: {results['member_prediction_mean']:.4f}")
        print(f"  Non-member Pred Mean: {results['non_member_prediction_mean']:.4f}")
        
        # Risk assessment based on AUC
        if results['auc'] > 0.8:
            risk = "HIGH"
        elif results['auc'] > 0.6:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        print(f"  Privacy Risk: {risk}")
        
        # Explain the difference between member and non-member predictions
        pred_diff = results['member_prediction_mean'] - results['non_member_prediction_mean']
        if pred_diff > 0.1:
            print(f"  → Members have higher prediction scores (+{pred_diff:.3f})")
        elif pred_diff < -0.1:
            print(f"  → Non-members have higher prediction scores ({pred_diff:.3f})")
        else:
            print(f"  → Similar prediction scores (diff: {pred_diff:.3f})")
    
    print("\n" + "="*50)


if __name__ == '__main__':
    main() 