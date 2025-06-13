import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class PopulationAttack:
    """
    Population-based Membership Inference Attack.
    
    This attack uses population statistics to determine membership.
    It compares model predictions against a baseline distribution
    computed from population data.
    """
    
    def __init__(self, device: str = 'cpu', method: str = "average_distribution"):
        """
        Initialize Population Attack.
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
            method: Method to compute baseline ("average_distribution" or "average_logits")
        """
        self.device = device
        self.method = method
        self.baseline_dist = None
        
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
                # default to KL if not recognized
                kl = np.sum(p * np.log((p + epsilon) / (baseline_dist + epsilon)))
                dist_or_score = kl
                membership_prediction = 1 if kl < threshold else 0

            results.append({
                "sample_index": sample_index,
                "membership_prediction": membership_prediction,
                "distance_or_score": dist_or_score,
            })
            sample_index += 1
    df_results = pd.DataFrame(results)
    return df_results