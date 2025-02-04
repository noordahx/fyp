# logic for training shadow models

import numpy as np
import os
import torch
from torch.utils.data import Subset
from mia_lib.trainer import train_model
from mia_lib.models import create_model
from mia_lib.data import create_subset_dataloader

def train_shadow_models(config, full_dataset, device):
    """
    Creates N shadow models according to config,
    trains each on a random subet of the dataset,
    and returns references to the trained shadow models (or just paths).
    """
    num_shadow = config["shadow"]["num_shadow_models"]
    shadow_train_size = config["shadow"]["shadow_train_size"]
    shadow_eval_size = config["shadow"]["shadow_eval_size"]
    shadow_test_size = config["shadow"]["shadow_test_size"]

    model_save_dir = config["paths"]["shadow_model_save_dir"]
    os.makedirs(model_save_dir, exist_ok=True)

    shadow_indices_save_dir = config["paths"]["shadow_indices_save_dir"]
    os.makedirs(shadow_indices_save_dir, exist_ok=True)

    shadow_models = []
    indices = np.arange(len(full_dataset))

    for i in range(num_shadow):
        # Decide wheter to load or train shadow indices
        shadow_indices_path = os.path.join(shadow_indices_save_dir, f"shadow_indices_{i}.npz")

        if os.path.exists(shadow_indices_path):
            data = np.load(shadow_indices_path)
            train_idx, eval_idx, test_idx = data["train_idx"], data["eval_idx"], data["test_idx"]
            print(f"[Shadow {i}] Loaded existing indices from {shadow_indices_path}")
        else:
                
            train_idx = np.random.choice(indices, shadow_train_size, replace=False)
            remaining = np.setdiff1d(indices, train_idx)
            eval_idx = np.random.choice(remaining, shadow_eval_size, replace=False)
            remaining = np.setdiff1d(remaining, eval_idx)
            test_idx = np.random.choice(remaining, shadow_test_size, replace=False)
            np.savez(shadow_indices_path, train_idx=train_idx, eval_idx=eval_idx, test_idx=test_idx)
            print(f"[Shadow {i}] Saved shadow indices to {shadow_indices_path}")
        
        train_loader = create_subset_dataloader(
            full_dataset,
            train_idx,
            batch_size=config["dataset"]["train_batch_size"],
            shuffle=True,
            num_workers=config["dataset"]["num_workers"]
        )

        eval_loader = create_subset_dataloader(
            full_dataset,
            eval_idx,
            batch_size=config["dataset"]["eval_batch_size"],
            shuffle=False,
            num_workers=config["dataset"]["num_workers"]
        )

        # Create the model
        save_path = os.path.join(model_save_dir, f"shadow_model_{i}.pth")
        shadow_model = create_model(config).to(device)

        if os.path.exists(save_path):
            print(f"[Shadow {i}] Found checkpoint at {save_path}. Loading...")
            shadow_model.load_state_dict(torch.load(save_path, map_location=device))
        else:
            print(f"[Shadow {i}] No checkpoint found at {save_path}. Training shadow model {i}...")
            train_model(shadow_model, train_loader, eval_loader, config, device, save_path)

        shadow_models.append((shadow_model, train_idx, eval_idx, test_idx))

    return shadow_models