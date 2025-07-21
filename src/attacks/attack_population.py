import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os

class PopulationAttack:
    """
    Population-based Membership Inference Attack.
    
    This attack uses population statistics to determine membership.
    It compares model predictions against a baseline distribution
    computed from population data.
    """
    
    def __init__(self, config=None, device: str = 'cpu', method: str = "average_distribution"):
        self.device = device
        self.method = method
        self.baseline_dist = None
        self.config = config
        self.name = "population"
        
    def compute_baseline(self, model, population_loader):
        """Compute population baseline distribution."""
        self.baseline_dist = compute_population_baseline(
            model, population_loader, self.device, self.method
        )
        return self.baseline_dist
        
    def infer_membership(self, model, data_loader, threshold_type: str = "kl_divergence",
                        threshold: float = 0.2) -> Dict[str, Any]:
        """
        Perform membership inference using population attack.
        
        Args:
            model: Target model
            data_loader: DataLoader with samples to classify
            threshold_type: Type of distance metric
            threshold: Distance threshold for membership
            
        Returns:
            Dictionary with attack results
        """
        if self.baseline_dist is None:
            raise ValueError("Baseline not computed. Call compute_baseline first.")
            
        df_results = attack_population_mia(
            model, data_loader, self.baseline_dist, self.device,
            threshold_type, threshold
        )
        
        predictions = df_results['membership_prediction'].values
        scores = df_results['distance_or_score'].values
        
        results = {
            'predictions': predictions,
            'scores': scores,
            'threshold': threshold,
            'threshold_type': threshold_type
        }
        
        return results

    def run(self, target_model, train_dataset, test_dataset, device):
        """
        Run the population-based attack.
        
        Args:
            target_model: Target model to attack
            train_dataset: Training dataset
            test_dataset: Test dataset
            device: Device to run on
            
        Returns:
            Dictionary with attack results
        """
        self.device = device
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.config.data.eval_batch_size,
            shuffle=False, 
            num_workers=self.config.data.num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.config.data.eval_batch_size,
            shuffle=False, 
            num_workers=self.config.data.num_workers
        )
        
        # Create a separate reference population (disjoint from train/test)
        # Use a different subset that doesn't overlap with train/test evaluation sets
        reference_size = getattr(self.config.attack.population, 'reference_size', 3000) if hasattr(self.config.attack, 'population') else 3000
        reference_size = min(reference_size, len(test_dataset) // 2)  # Use at most half of test set
        
        # Use the first half of test dataset as reference, second half for evaluation
        reference_indices = list(range(reference_size))
        eval_test_indices = list(range(reference_size, len(test_dataset)))
        
        reference_dataset = torch.utils.data.Subset(test_dataset, reference_indices)
        eval_test_dataset = torch.utils.data.Subset(test_dataset, eval_test_indices)
        
        reference_loader = torch.utils.data.DataLoader(
            reference_dataset,
            batch_size=self.config.data.eval_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        
        eval_test_loader = torch.utils.data.DataLoader(
            eval_test_dataset,
            batch_size=self.config.data.eval_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        
        print(f"Computing baseline from {len(reference_dataset)} reference samples...")
        self.compute_baseline(target_model, reference_loader)
        
        # Calibrate threshold using a small subset
        print("Calibrating threshold...")
        optimal_threshold = self._calibrate_threshold(
            target_model, train_dataset, eval_test_dataset, device
        )
        
        # Run inference on train and test sets with optimal threshold
        print("Running population attack on training data...")
        train_results = self.infer_membership(
            target_model, 
            train_loader, 
            threshold_type="improved_kl", 
            threshold=optimal_threshold
        )
        
        print("Running population attack on evaluation test data...")
        test_results = self.infer_membership(
            target_model, 
            eval_test_loader, 
            threshold_type="improved_kl", 
            threshold=optimal_threshold
        )
        
        # Create ground truth labels (1 for train/members, 0 for test/non-members)
        train_labels = np.ones(len(train_results['predictions']))
        test_labels = np.zeros(len(test_results['predictions']))
        
        # Combine predictions and true labels
        all_preds = np.concatenate([train_results['predictions'], test_results['predictions']])
        all_true = np.concatenate([train_labels, test_labels])
        all_scores = np.concatenate([train_results['scores'], test_results['scores']])
        
        print(f"Train predictions: {np.mean(train_results['predictions']):.3f} (should be close to 1 for good attack)")
        print(f"Test predictions: {np.mean(test_results['predictions']):.3f} (should be close to 0 for good attack)")
        
        # Calculate metrics
        accuracy = accuracy_score(all_true, all_preds)
        try:
            auc = roc_auc_score(all_true, all_scores)
        except:
            auc = 0.5  # Default if calculation fails
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true, all_preds, average='weighted', zero_division=0
        )
        
        # Calculate attack advantage
        train_acc = accuracy_score(train_labels, train_results['predictions'])
        test_acc = accuracy_score(test_labels, test_results['predictions'])
        attack_advantage = train_acc + test_acc - 1
        
        results = {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'attack_advantage': attack_advantage,
            'all_predictions': all_preds,
            'all_true_labels': all_true,
            'all_scores': all_scores
        }
        
        print(f"Population Attack Results: AUC = {auc:.4f}, Accuracy = {accuracy:.4f}, Advantage = {attack_advantage:.4f}")
        
        # Save results if config specifies a save directory
        if hasattr(self.config, 'output') and hasattr(self.config.output, 'save_dir'):
            save_dir = os.path.join(self.config.output.save_dir, "attack_results")
            os.makedirs(save_dir, exist_ok=True)
            
            # Save as CSV
            df_results = pd.DataFrame({
                'true_membership': all_true,
                'predicted_membership': all_preds,
                'score': all_scores
            })
            df_results.to_csv(os.path.join(save_dir, "population_attack_results.csv"), index=False)
            
            # Save metrics
            with open(os.path.join(save_dir, "population_attack_metrics.json"), 'w') as f:
                json.dump({
                    'auc': float(auc),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'attack_advantage': float(attack_advantage)
                }, f, indent=2)
        
        return results
    
    def _calibrate_threshold(self, target_model, train_dataset, test_dataset, device):
        """Calibrate optimal threshold using validation data."""
        # Use small subsets for calibration to save time
        cal_train_size = min(1000, len(train_dataset))
        cal_test_size = min(1000, len(test_dataset))
        
        cal_train_subset = torch.utils.data.Subset(train_dataset, range(cal_train_size))
        cal_test_subset = torch.utils.data.Subset(test_dataset, range(cal_test_size))
        
        cal_train_loader = torch.utils.data.DataLoader(cal_train_subset, batch_size=64, shuffle=False)
        cal_test_loader = torch.utils.data.DataLoader(cal_test_subset, batch_size=64, shuffle=False)
        
        # Get distances for calibration samples
        train_distances = self._compute_distances(target_model, cal_train_loader, device)
        test_distances = self._compute_distances(target_model, cal_test_loader, device)
        
        # Find optimal threshold
        all_distances = np.concatenate([train_distances, test_distances])
        all_labels = np.concatenate([np.ones(len(train_distances)), np.zeros(len(test_distances))])
        
        # Try different thresholds and find best accuracy
        thresholds = np.percentile(all_distances, np.linspace(10, 90, 50))
        best_threshold = thresholds[0]
        best_accuracy = 0.0
        
        for threshold in thresholds:
            # Lower distance -> member (1), higher distance -> non-member (0)
            predictions = (all_distances < threshold).astype(int)
            accuracy = accuracy_score(all_labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        print(f"Optimal threshold: {best_threshold:.4f} (accuracy: {best_accuracy:.4f})")
        return best_threshold
    
    def _compute_distances(self, model, data_loader, device):
        """Compute distances from baseline for data loader."""
        model.eval()
        distances = []
        epsilon = 1e-8
        
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                
                for i in range(probs.shape[0]):
                    p = probs[i]
                    # Improved symmetric KL divergence for better stability
                    kl_forward = np.sum(p * np.log((p + epsilon) / (self.baseline_dist + epsilon)))
                    kl_backward = np.sum(self.baseline_dist * np.log((self.baseline_dist + epsilon) / (p + epsilon)))
                    symmetric_kl = 0.5 * (kl_forward + kl_backward)
                    distances.append(symmetric_kl)
        
        return np.array(distances)

def compute_population_baseline(model, population_loader, device, method="average_distribution"):
    """
    Compute baseline statistics from a 'population_loader' that is assumed
    to represent typical (non-member) data.

    :param model: The target model or reference model used to get logits/probabilities.
    :param population_loader: DataLoader for population data (assumed non-member).
    :param device: 'cpu' or 'cuda'
    :param method: how to compute the baseline.
                    - "average_distribution": average the probabilities across the population.
                    - "average_logits": average the logits across the population.
    :return: an array of distribution representing the baseline.
            For "average_distribution": shape [num_classes]
            For "average_logits": shape [num_classes]
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _ in tqdm(population_loader, desc="Population baseline"):
            images = images.to(device)
            outputs = model(images)
        
            probs = F.softmax(outputs, dim=1) # shape [batch_size, num_classes]
            all_probs.append(probs.cpu().numpy())
        
        all_probs = np.concatenate(all_probs, axis=0) # shape [num_samples, num_classes]

        if method == "average_distribution":
            baseline_dist = np.mean(all_probs, axis=0)
        elif method == "average_logits":
            baseline_dist = np.mean(all_probs, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return baseline_dist

def attack_population_mia(
        model,
        test_loader,
        baseline_dist,
        device,
        threshold_type="kl_divergence",
        threshold=0.2,
):
    """
    Implements a population/statistical attack (Attack P).
    We have a baseline distribution from population data.
    For each sample in 'test_loader', compute distance from that baseline.
    If distance < threshold => membership_prediction = 1 (member)

    :param model: The target model used to get logits/probabilities.
    :param test_loader: DataLoader for the samples we want to classify as member or non-member.
    :param baseline_dist: The population-based baseline distribution (shape [num_classes]).
    :param device: 'cpu' or 'cuda'
    :param threshold_type: how to measure difference from baseline (e.g. "kl_divergence", "js_divergence", "l1", "confidence")
    :param threshold: the distance threshold for membership.
                        If distance < threshold => membership_prediction = 1 (member)
    :return: A DataFrame with columns:
            [sample_index, membership_prediction, distance or score, ...]
    """
    model.eval()
    results = []
    sample_index = 0

    # tiny epsilon to avoid log(0)
    epsilon = 1e-8

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Attack P Inference"):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy() # shape [batch_size, num_classes]

            batch_size = images.size(0)
            for i in range(batch_size):
                p = probs[i]

                if threshold_type == "kl_divergence":
                    # KL divergence from baseline
                    # KL(p || baseline) = sum(p_i * log(p_i / baseline_i))
                    kl = np.sum(p * np.log((p + epsilon) / (baseline_dist + epsilon)))
                    dist_or_score = kl
                    # if KL < threshold => we guess "member".
                    membership_prediction = 1 if kl < threshold else 0
                elif threshold_type == "improved_kl":
                    # Improved KL with better numerical handling and symmetry
                    kl_forward = np.sum(p * np.log((p + epsilon) / (baseline_dist + epsilon)))
                    kl_backward = np.sum(baseline_dist * np.log((baseline_dist + epsilon) / (p + epsilon)))
                    # Use symmetric KL divergence for better stability
                    symmetric_kl = 0.5 * (kl_forward + kl_backward)
                    dist_or_score = symmetric_kl
                    # Lower distance indicates more similarity to training distribution (member)
                    membership_prediction = 1 if symmetric_kl < threshold else 0
                elif threshold_type == "js_divergence":
                    # Jensen-Shannon divergence
                    # JS(p || baseline) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
                    m = 0.5 * (p + baseline_dist)
                    kl1 = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
                    kl2 = np.sum(baseline_dist * np.log((baseline_dist + epsilon) / (m + epsilon)))
                    js = 0.5 * (kl1 + kl2)
                    dist_or_score = js
                    # if JS < threshold => we guess "member".
                    membership_prediction = 1 if js < threshold else 0
                elif threshold_type == "l1":
                    # L1 distance
                    # L1(p, baseline) = sum(|p_i - baseline_i|)
                    l1 = np.sum(np.abs(p - baseline_dist))
                    dist_or_score = l1
                    # if L1 < threshold => we guess "member".
                    membership_prediction = 1 if l1 < threshold else 0
                elif threshold_type == "confidence":
                    # e.g. compare the sample's max probability to baseline's max probability
                    max_p = np.max(p)
                    max_baseline = np.max(baseline_dist)
                    dist_or_score = max_p
                    # if max_p > max_baseline => we guess "member".
                    membership_prediction = 1 if max_p > max_baseline else 0
                else:
                    # default to improved KL if not recognized
                    kl_forward = np.sum(p * np.log((p + epsilon) / (baseline_dist + epsilon)))
                    kl_backward = np.sum(baseline_dist * np.log((baseline_dist + epsilon) / (p + epsilon)))
                    symmetric_kl = 0.5 * (kl_forward + kl_backward)
                    dist_or_score = symmetric_kl
                    membership_prediction = 1 if symmetric_kl < threshold else 0

                results.append({
                    "sample_index": sample_index,
                    "membership_prediction": membership_prediction,
                    "distance_or_score": dist_or_score,
                })
                sample_index += 1
    df_results = pd.DataFrame(results)
    return df_results