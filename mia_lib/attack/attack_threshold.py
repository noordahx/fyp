"""
Threshold-based Membership Inference Attack.

This attack uses confidence thresholds to determine membership.
If a model's confidence on a sample is above a threshold, it's classified as a member.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict, Any, Tuple, Optional
import pandas as pd


class ThresholdAttack:
    """
    Threshold-based Membership Inference Attack.
    
    This attack assumes that models have higher confidence on training data
    (members) compared to test data (non-members).
    """
    
    def __init__(self, threshold: float = 0.5, device: str = 'cpu'):
        """
        Initialize Threshold Attack.
        
        Args:
            threshold: Confidence threshold for membership classification
            device: Device to run on ('cpu' or 'cuda')
        """
        self.threshold = threshold
        self.device = device
        self.optimal_threshold = None
        
    def calibrate_threshold(self, model, member_loader, non_member_loader) -> float:
        """
        Calibrate the optimal threshold using validation data.
        
        Args:
            model: Target model
            member_loader: DataLoader with member samples
            non_member_loader: DataLoader with non-member samples
            
        Returns:
            Optimal threshold value
        """
        model.eval()
        confidences = []
        labels = []
        
        with torch.no_grad():
            # Get confidences for members
            for images, _ in tqdm(member_loader, desc="Processing members"):
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                confidences.extend(max_probs.cpu().numpy())
                labels.extend([1] * len(images))
                
            # Get confidences for non-members
            for images, _ in tqdm(non_member_loader, desc="Processing non-members"):
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                confidences.extend(max_probs.cpu().numpy())
                labels.extend([0] * len(images))
        
        # Find optimal threshold
        confidences = np.array(confidences)
        labels = np.array(labels)
        
        # Print statistics for debugging
        member_confs = confidences[labels == 1]
        non_member_confs = confidences[labels == 0]
        print(f"Member confidence - Mean: {member_confs.mean():.4f}, Std: {member_confs.std():.4f}")
        print(f"Non-member confidence - Mean: {non_member_confs.mean():.4f}, Std: {non_member_confs.std():.4f}")
        
        best_threshold = 0.5
        best_accuracy = 0.0
        
        # Try different thresholds based on percentiles
        thresholds = np.percentile(confidences, np.linspace(10, 90, 100))
        for threshold in thresholds:
            predictions = (confidences > threshold).astype(int)
            accuracy = accuracy_score(labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                
        self.optimal_threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.4f}, Accuracy: {best_accuracy:.4f}")
        return best_threshold
        
    def infer_membership(self, model, data_loader, 
                        use_optimal_threshold: bool = True) -> Dict[str, Any]:
        """
        Perform membership inference using threshold attack.
        
        Args:
            model: Target model
            data_loader: DataLoader with samples to classify
            use_optimal_threshold: Whether to use calibrated threshold
            
        Returns:
            Dictionary with attack results
        """
        threshold = self.optimal_threshold if (use_optimal_threshold and self.optimal_threshold) else self.threshold
        
        model.eval()
        confidences = []
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Threshold attack inference"):
                if len(batch) == 2:
                    images, labels = batch
                else:
                    # Handle case where batch might have different structure
                    images = batch[0]
                    labels = batch[1] if len(batch) > 1 else torch.zeros(len(images))
                
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                
                confidences.extend(max_probs.cpu().numpy())
                batch_predictions = (max_probs.cpu().numpy() > threshold).astype(int)
                predictions.extend(batch_predictions)
                true_labels.extend(labels.numpy())
        
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions) if len(set(true_labels)) > 1 else 0.0
        
        results = {
            'predictions': predictions,
            'confidences': confidences,
            'probabilities': confidences,  # For visualization compatibility
            'true_labels': true_labels,    # For visualization compatibility
            'threshold': threshold,
            'accuracy': accuracy
        }
        
        # Calculate additional metrics if we have both classes
        if len(set(true_labels)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary', zero_division=0
            )
            try:
                auc_score = roc_auc_score(true_labels, confidences)
            except ValueError as e:
                print(f"AUC calculation failed: {e}")
                auc_score = 0.5
            
            results.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score
            })
        
        return results
        
    def analyze_confidence_distribution(self, model, member_loader, 
                                      non_member_loader) -> Dict[str, np.ndarray]:
        """
        Analyze confidence distributions for members vs non-members.
        
        Args:
            model: Target model
            member_loader: DataLoader with member samples
            non_member_loader: DataLoader with non-member samples
            
        Returns:
            Dictionary with confidence distributions
        """
        model.eval()
        member_confidences = []
        non_member_confidences = []
        
        with torch.no_grad():
            # Get member confidences
            for images, _ in tqdm(member_loader, desc="Analyzing member confidences"):
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                member_confidences.extend(max_probs.cpu().numpy())
                
            # Get non-member confidences
            for images, _ in tqdm(non_member_loader, desc="Analyzing non-member confidences"):
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                non_member_confidences.extend(max_probs.cpu().numpy())
        
        return {
            'member_confidences': np.array(member_confidences),
            'non_member_confidences': np.array(non_member_confidences),
            'member_mean': np.mean(member_confidences),
            'non_member_mean': np.mean(non_member_confidences),
            'member_std': np.std(member_confidences),
            'non_member_std': np.std(non_member_confidences)
        } 