import os
import numpy as np
import torch
from torch.utils.data import Subset
from mia_lib.data import create_subset_dataloader
from mia_lib.models import create_model
from mia_lib.trainer import trainer
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

class ReferenceAttack:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.reference_models = []

    def train_reference_models(self, full_dataset, n_models=5):
        """
        Train or load a sot of 'reference models' for Attack R.
        Each reference model is trained on a (possibly) different subset
        of 'full_dataset' to approximate the distributeion of non-member data.

        :param full_dataset: a dataset (e.g., CIFAR-10) from which to sample training data
        :param n_models: how many reference models to train

        :return: a list of trained reference models
        """

        model_save_dir=self.config["paths"]["model_save_dir"]
        os.makedirs(model_save_dir, exist_ok=True)

        ref_models = []
        total_indices = np.arange(len(full_dataset))
        train_size = self.config["reference"].get("ref_train_size", 5000)

        for i in range(n_models):
            # 1) Generate or load existing subset indices
            ref_indices_path = os.path.join(model_save_dir, f"reference_{i}_indices.npz")
            if os.path.exists(ref_indices_path):
                data = np.load(ref_indices_path)
                ref_train_idx = data["ref_train_idx"]
                print(f"[Ref {i}] Loaded existing indices from {ref_indices_path}")
            else:
                # Create a new random subset
                ref_train_idx = np.random.choice(total_indices, train_size, replace=False)
                np.savez(ref_indices_path, ref_train_idx=ref_train_idx)
                print(f"[Ref {i}] Saved new indices to {ref_indices_path}")
            
            # 2) Build DataLoader
            ref_train_loader = create_subset_dataloader(
                full_dataset,
                ref_train_idx,
                batch_size=self.config["dataset"]["train_batch_size"],
                shuffle=True,
                num_workers=self.config["dataset"]["num_workers"]
            )

            # 3) Create or load reference model
            save_path = os.path.join(model_save_dir, f"reference_model_{i}.pth")
            ref_model = create_model(self.config).to(self.device)

            if os.path.exists(save_path):
                # Load existing checkpoint
                print(f"[Ref {i}] Found checkpoint at {save_path}. Loading...")
                ref_model.load_state_dict(torch.load(save_path, map_location=self.device))
            else:
                print(f"[Ref {i}] No checkpoint found at {save_path}. Training new reference model...")
                # Train
                # We can optionally re-use the same 'train_model' function as the target/shadow
                # For validation, pick another random subset or do a simple holdout
                trainer(ref_model, ref_train_loader, ref_train_loader, self.config, self.device, save_path)
                # if separate validation set exists, pass it instead of ref_train_loader for validation.
            ref_models.append(ref_model)
        
        return ref_models



    def attack_reference_mia(
            self,
            target_model,
            reference_models,
            loader,
            method="mean_prob_distance",
            threshold=0.5
    ):
        """
        Attack R: sample-dependent MIA via Reference models.

        1) Compute the target model's probability distribution for each sample in 'loader'.
        2) Compute the reference models' probability distributions for each sample,
            then average them (or pick your own aggregation).
        3) Compare target_probs vs. ref_probs with distance metric (L1, L2, or others).
        4) If disatnece < threshold => guess "member", else "non-member".
        (You might invert the inequality depending on your logic).

        :param target_model: The main model we're attacking.
        :param reference_models: A list of reference models, each trained on some data subset.
        :param loader: DataLoader of the samples we want to classify as member or non-member.
        :param device: 'cpu' or 'cuda'
        :param method: how to measure difference from references (e.g. "mean_prob_distance")
        :param threshold: the distance threshold for membership.
                            If distance < threshold => membership_prediction = 1 (member)
        
        :return: A DataFrame with columns:
                [sample_index, membership_prediction, distance_metric, ...]
        """

        target_model.eval()
        for ref_model in reference_models:
            ref_model.eval()
        
        results = []
        sample_index = 0

        with torch.no_grad():
            for images, _ in tqdm(loader, desc="Attack R Inference"):
                images = images.to(self.device)

                # 1) Target model probabilities
                target_outputs = target_model(images)
                target_probs = F.softmax(target_outputs, dim=1).cpu().numpy() # shape [batch_size, num_classes]

                # 2) Reference ensemble probabilities
                ref_probs_list = []
                for ref_model in reference_models:
                    ref_out = ref_model(images)
                    ref_probs = F.softmax(ref_out, dim=1).cpu().numpy() # shape [batch_size, num_classes]
                    ref_probs_list.append(ref_probs)
                
                # shape => [num_ref_models, batch_size, num_classes]
                ref_probs_array = np.stack(ref_probs_list, axis=0)

                # 3) Aggregate (e.g. average across reference models)
                # shape => [batch_size, num_classes]
                avg_ref_probs = np.mean(ref_probs_array, axis=0)

                # 4) For each sample in the batch
                batch_size = target_probs.shape[0]
                for i in range(batch_size):
                    t_prob = target_probs[i]        # shape[num_classes]
                    r_prob = avg_ref_probs[i]       # shape[num_classes]
                
                    # Example distance measure:
                    # L1 distance
                    if method == "mean_prob_distance":
                        # formula: mean(|p_i - q_i|)
                        dist = np.sum(np.abs(t_prob - r_prob)) # L1
                    elif method == "l2":
                        # formula: sqrt(sum((p_i - q_i)^2))
                        dist = np.sqrt(np.sum((t_prob - r_prob) ** 2)) # L2
                    elif method == "KL":
                        # formula: sum(p_i * log(p_i / q_i))
                        dist = np.sum(t_prob * np.log(t_prob / r_prob))
                    elif method == "JS":
                        # formula: 0.5 * (KL(p||m) + KL(q||m))
                        m = 0.5 * (t_prob + r_prob)
                        dist = 0.5 * np.sum(t_prob * np.log(t_prob / m)) + 0.5 * np.sum(r_prob * np.log(r_prob / m))
                    else:
                        print(f"Unknown distance method: {method}, using L1 by default.")
                        dist = np.sum(np.abs(t_prob - r_prob)) # L1

                    # 5) membership prediction based on threshold 
                    #  If we assume smaller distance => the target model is "similar" to reference
                    #  => more likely to be a "member".
                    # (might be inverted if needed)
                    membership_pred = 1 if dist < threshold else 0

                    results.append({
                        "sample_index": sample_index,
                        "membership_prediction": membership_pred,
                        "distance_metric": float(dist)
                    })      
                    sample_index += 1
            df_results = pd.DataFrame(results)
            return df_results
