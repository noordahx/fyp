import os
import numpy as np
import pandas as pd
from src.new_utils.models import create_model
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

from src.new_utils.data import create_subset_dataloader


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
    """
    base_model.eval()

    # If no indices provided, pick a small random subset
    if indices_to_test is None:
        indices_to_test = np.random.choice(len(full_dataset), 10, replace=False)
    
    results = []
    sample_counter = 0

    # Assume the entire dataset was used for training
    original_indices = np.arange(len(full_dataset))

    # Handle epochs for retraining
    original_epochs = config.training.epochs
    if retrain_epochs is not None:
        # We'll save and restore the original epochs
        current_epochs = original_epochs
        config.training.epochs = retrain_epochs
    
    # For each sample we want to test
    for sample_idx in indices_to_test:
        # Create training subset without x_i
        loo_train_indices = np.setdiff1d(original_indices, [sample_idx])
        train_loader = create_subset_dataloader(
            full_dataset,
            loo_train_indices,
            batch_size=config.data.train_batch_size,   
            shuffle=True,
            num_workers=config.data.num_workers
        )
    
        # Init a new model
        model_without_x = create_model(config).to(device)

        # Retrain the model without this sample
        save_path = os.path.join(
            config.output.save_dir,
            f"loo_temp_model_{sample_idx}.pth"
        )
        
        from src.attacks.trainer import trainer
        trainer(
            model_without_x,
            train_loader,
            train_loader,
            config,
            save_path
        )

        # Evaluate difference for sample x_i
        image, label = full_dataset[sample_idx]
        image = image.unsqueeze(0).to(device)

        base_out = base_model(image)
        model_without_x_out = model_without_x(image)

        # Measure difference in output probabilities
        base_probs = F.softmax(base_out, dim=1)
        loo_probs = F.softmax(model_without_x_out, dim=1)
        diff_value = torch.sum(torch.abs(base_probs - loo_probs)).item()

        # Membership logic: if diff is large => likely "member"
        membership_prediction = 1 if diff_value > threshold else 0

        results.append({
            "sample_index": sample_idx,
            "membership_prediction": membership_prediction,
            "diff_value": diff_value
        })

        sample_counter += 1
    
    # Restore original epochs
    if retrain_epochs is not None:
        config.training.epochs = original_epochs

    df_loo = pd.DataFrame(results)
    return df_loo
