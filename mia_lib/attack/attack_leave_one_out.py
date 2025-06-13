import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List
from sklearn.metrics import accuracy_score

from mia_lib.trainer import trainer
from mia_lib.models import create_model
from mia_lib.data import create_subset_dataloader


class LeaveOneOutAttack:
    """
    Leave-One-Out Membership Inference Attack.
    
    This attack retrains models without each sample and measures
    the difference in predictions to infer membership.
    WARNING: Very computationally expensive!
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """
        Initialize Leave-One-Out Attack.
        
        Args:
            config: Configuration dictionary
            device: Device to run on ('cpu' or 'cuda')
        """
        self.config = config
        self.device = device
        
    def infer_membership(self, base_model, full_dataset, 
                        indices_to_test: Optional[List[int]] = None,
                        threshold: float = 0.05,
                        retrain_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform membership inference using leave-one-out attack.
        
        Args:
            base_model: Original model trained on full dataset
            full_dataset: Full dataset used for training
            indices_to_test: Indices to test (if None, uses small random subset)
            threshold: Difference threshold for membership
            retrain_epochs: Number of epochs for retraining
            
        Returns:
            Dictionary with attack results
        """
        df_results = leave_one_out_attack(
            self.config, full_dataset, base_model, self.device,
            indices_to_test, threshold, retrain_epochs
        )
        
        predictions = df_results['membership_prediction'].values
        diff_values = df_results['diff_value'].values
        
        results = {
            'predictions': predictions,
            'diff_values': diff_values,
            'threshold': threshold,
            'tested_indices': df_results['sample_index'].values
        }
        
        return results

def leave_one_out_attack(
        config,
        full_dataset,
        base_model,
        device,
        indices_to_test=None,
        threshold=0.05,
        retrain_epochs=None
):
    """
    Naive Leave-One-Out Attack (Attack L).

    Steps for each sample x_i in 'indices_to_test':
    1. Remove x_i from training set.
    2. Retrain (or partially retrain) a 'model_without_x'.
    3. Measure difference in predictions for x_i:
        diff = distance(base_model(x_i), model_without_x(x_i))
    4. If diff < threshold => 'member' else 'non-member'

    WARNING: Very expensive if you do this for many samples.
            Typically feasible only for a small subset or with approximations.
    :param config: global config dict
    :param full_dataset: the original dataset used to train 'base_model'
    :param base_model: the original model trained on the entire dataset (the model under attack)
    :param device: 'cpu' or 'cuda'
    :param indices_to_test: indices of samples to test (if None, pick a small random subset)
    :param threshold: difference threshold above wihch we guess 'member'
    :param retrain_epochs: optionally override config["training"]["epochs"] for speed.
                            if None, uses config as it is.
    :return: DataFrame with columns:
            [sample_index, membership_prediction, diff_value]
    """
    base_model.eval()

    # If no indices provided, pick a small random subset (e.g. 10 samples) for demo
    if indices_to_test is None:
        indices_to_test = np.random.choice(len(full_dataset), 10, replace=False)
    
    # We'll store results (sample index, membership_prediction, difference)
    results = []
    sample_counter = 0

    # Original training set indices, if known.
    # For demo, let's assume the entier 'full_dataset' was used for training.
    # In a real scenario, might have a specialized subset or config param
    # storing the 'target' training indices.
    original_indices = np.arange(len(full_dataset))

    # We'll see if user wants fewer epochs for partial training
    original_epochs = config["training"].get("epochs", 10)
    if retrain_epochs is not None:
        config["training"]["epochs"] = retrain_epochs
    
    # for each sample we want to test:
    for sample_idx in indices_to_test:
        # 1) Create training subset without x_i
        loo_train_indices = np.setdiff1d(original_indices, [sample_idx])
        train_loader = create_subset_dataloader(
            full_dataset,
            loo_train_indices,
            batch_size=config["dataset"]["train_batch_size"],   
            shuffle=True,
            num_workers=config["dataset"]["num_workers"]
        )
    
        # 2) Init a new model
        model_without_x = create_model(config).to(device)

        # 3) Retrain or partially retrain the model_without_x
        #   We'll reuse 'train_model' function, but pass the same loader for train/eval
        #   or create your own val subset
        #   to keep it short.
        save_path = os.path.join(
            config["paths"]["model_save_dir"],
            f"loo_temp_model_{sample_idx}.pth"
        )
        trainer(
            model_without_x,
            train_loader,
            train_loader,
            config,
            device,
            save_path,
        )

        # 4) Evaluate difference for sample x_i
        #    We'll just do a single pass:
        image, label = full_dataset[sample_idx]
        image = image.unsqueeze(0).to(device) # [1, C, H, W]

        base_out = base_model(image)
        model_without_x_out = model_without_x(image)

        # Let's measure teh L2 diff in logits or some other measure
        # Can also measure diff in predicted probs
        base_probs = F.softmax(base_out, dim=1)
        loo_probs = F.softmax(model_without_x_out, dim=1)

        # L1 diff
        diff_value = torch.sum(torch.abs(base_probs - loo_probs)).item()

        # membership logic: if diff is large => the sample significantly impacted 
        # the original model => more likely "member"
        membership_prediction = 1 if diff_value < threshold else 0

        # store result
        results.append({
            "sample_index": sample_idx,
            "membership_prediction": membership_prediction,
            "diff_value": diff_value
        })

        sample_counter += 1
    
    # Restore original epochs
    config["training"]["epochs"] = original_epochs

    df_loo = pd.DataFrame(results)
    return df_loo
