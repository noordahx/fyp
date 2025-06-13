# logic for training shadow models

import numpy as np
import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Subset
import torch.nn.functional as F
from mia_lib.data import create_subset_dataloader
from mia_lib.trainer import trainer
from mia_lib.models import create_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from typing import List, Tuple, Dict, Any, Optional
import torch.nn as nn


class ShadowAttack:
    """
    Shadow Model Attack for Membership Inference.
    
    This attack trains multiple shadow models on synthetic data to learn
    the behavior of models on member vs non-member data, then uses this
    knowledge to attack the target model.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """
        Initialize Shadow Attack.
        
        Args:
            config: Configuration dictionary
            device: Device to run on ('cpu' or 'cuda')
        """
        self.config = config
        self.device = device
        self.shadow_models = []
        self.attack_model = None
        
    def train_shadow_models(self, full_dataset) -> List[Tuple]:
        """Train shadow models and return model info."""
        return train_shadow_models(self.config, full_dataset)
        
    def create_attack_dataset(self, shadow_models_info: List[Tuple], 
                            full_dataset) -> pd.DataFrame:
        """
        Create attack dataset from shadow models.
        
        Args:
            shadow_models_info: List of (model, train_idx, eval_idx, test_idx)
            full_dataset: Full dataset
            
        Returns:
            DataFrame with attack features and labels
        """
        attack_datasets = []
        
        for i, (shadow_model, train_idx, eval_idx, test_idx) in enumerate(shadow_models_info):
            print(f"Creating attack dataset from shadow model {i}")
            
            # Create data loaders
            train_loader = create_subset_dataloader(
                full_dataset, train_idx,
                batch_size=self.config.dataset.eval_batch_size,
                shuffle=False, num_workers=self.config.dataset.num_workers
            )
            test_loader = create_subset_dataloader(
                full_dataset, test_idx,
                batch_size=self.config.dataset.eval_batch_size,
                shuffle=False, num_workers=self.config.dataset.num_workers
            )
            
            # Create attack dataset for this shadow model
            df_attack = create_shadow_attack_dataset(
                shadow_model, train_loader, test_loader, self.device
            )
            attack_datasets.append(df_attack)
            
        # Combine all attack datasets
        combined_df = pd.concat(attack_datasets, ignore_index=True)
        return combined_df
        
    def train_attack_model(self, df_attack: pd.DataFrame):
        """Train the attack model."""
        self.attack_model = train_attack_model(df_attack, self.config)
        return self.attack_model
        
    def infer_membership(self, target_model, target_data_loader) -> Dict[str, float]:
        """
        Perform membership inference on target model.
        
        Args:
            target_model: Target model to attack
            target_data_loader: Data loader with target data
            
        Returns:
            Dictionary with attack results
        """
        if self.attack_model is None:
            raise ValueError("Attack model not trained. Call train_attack_model first.")
            
        # Get target model predictions
        target_model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(target_data_loader, desc="Getting target predictions"):
                images = images.to(self.device)
                outputs = target_model(images)
                probs = F.softmax(outputs, dim=1)
                top_p, _ = probs.topk(10, dim=1)
                predictions.append(top_p.cpu().numpy())
                true_labels.extend(labels.numpy())
                
        predictions = np.concatenate(predictions, axis=0)
        
        # Create DataFrame for attack model
        columns = [f"top_{i}_prob" for i in range(10)]
        df_target = pd.DataFrame(predictions, columns=columns)
        
        # Get attack predictions
        attack_predictions = self.attack_model.predict(df_target)
        attack_probs = self.attack_model.predict_proba(df_target)[:, 1]
        
        return {
            'predictions': attack_predictions,
            'probabilities': attack_probs,
            'accuracy': accuracy_score(true_labels, attack_predictions) if len(set(true_labels)) > 1 else 0.0
        }

def train_shadow_models(CFG_ATTACK, full_dataset):
    """
    Trains (or loads) multiple shadow models, each on a random subset of 'full_dataset'.
    Returns a list of tuples: (shadow_model, train_indices, eval_indices, test_indices).
    
    Similar to other approach in Attack R or Attack P, but specifically for
    the shadow-model-based method of membership inference.

    :param config: dict containing hyperparameters and paths
    :param full_dataset: the dataset from which we draw subsets for shadow training
    :return: list of (model, train_idx, eval_idx, test_idx)
    """
    num_shadow = CFG_ATTACK.shadow.num_shadow_models
    shadow_train_size = CFG_ATTACK.shadow.shadow_train_size
    shadow_eval_size = CFG_ATTACK.shadow.shadow_eval_size
    shadow_test_size = CFG_ATTACK.shadow.shadow_test_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories for saving model checkpoints and indices
    model_save_dir = CFG_ATTACK.paths.attack_dir
    os.makedirs(model_save_dir, exist_ok=True)

    shadow_indices_save_dir = CFG_ATTACK.paths.attack_dir
    os.makedirs(shadow_indices_save_dir, exist_ok=True)

    indices = np.arange(len(full_dataset))
    shadow_models_info = []

    for i in range(num_shadow):
        # 1) Load or create the subset indices
        shadow_indices_path = os.path.join(shadow_indices_save_dir, f"shadow_indices_{i}.npz")

        if os.path.exists(shadow_indices_path):
            data = np.load(shadow_indices_path)
            train_idx, eval_idx, test_idx = data["train_idx"], data["eval_idx"], data["test_idx"]
            print(f"[Shadow {i}] Loaded existing indices from {shadow_indices_path}")
        else:
            # Randomly pick train, eval, test subsets
            train_idx = np.random.choice(indices, shadow_train_size, replace=False)
            remaining = np.setdiff1d(indices, train_idx)
            eval_idx = np.random.choice(remaining, shadow_eval_size, replace=False)
            remaining = np.setdiff1d(remaining, eval_idx)
            test_idx = np.random.choice(remaining, shadow_test_size, replace=False)

            np.savez(shadow_indices_path, train_idx=train_idx, eval_idx=eval_idx, test_idx=test_idx)
            print(f"[Shadow {i}] Saved new shadow indices to {shadow_indices_path}")

        # 2) Build DataLoaders
        train_loader = create_subset_dataloader(
            full_dataset,
            train_idx,
            batch_size=CFG_ATTACK.dataset.train_batch_size,
            shuffle=True,
            num_workers=CFG_ATTACK.dataset.num_workers
        )
        eval_loader = create_subset_dataloader(
            full_dataset,
            eval_idx,
            batch_size=CFG_ATTACK.dataset.eval_batch_size,
            shuffle=False,
            num_workers=CFG_ATTACK.dataset.num_workers
        )

        # 3) Create or load the shadow model
        save_path = os.path.join(model_save_dir, f"shadow_model_{i}.pth")
        
        # Create model using simple model creation
        shadow_model = _create_simple_model(CFG_ATTACK.model.num_classes).to(device)

        if os.path.exists(save_path):
            print(f"[Shadow {i}] Found checkpoint at {save_path}. Loading...")
            try:
                shadow_model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            except TypeError:
                shadow_model.load_state_dict(torch.load(save_path, map_location=device))
        else:
            print(f"[Shadow {i}] No checkpoint at {save_path}. Training new shadow model...")
            # Simple training loop for shadow model
            _train_shadow_model(shadow_model, train_loader, eval_loader, CFG_ATTACK, save_path, device)

        # Return model + subset indices for building the attack dataset
        shadow_models_info.append((shadow_model, train_idx, eval_idx, test_idx))

    return shadow_models_info

def _make_member_nonmember_dataset(model, train_loader, test_loader, device):
    """
    Helper for 'create_shadow_attack_dataset':
    - For each sample in 'member_loader', get top-k probs => label = 1
    - For each sample in 'nonmember_loader', get top-k probs => label = 0
    """
    model.eval()

    member_probs = []
    non_member_probs = []

    with torch.no_grad():
        # Member
        for images, _ in tqdm(train_loader, desc="Colecting MEMBER probabilities"):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            # For 10-class problem, topk(10) will basically returan all classes in sorted order
            top_p, _ = probs.topk(10, dim=1)
            member_probs.append(top_p.cpu().numpy())

        # Non-member
        for images, _ in tqdm(test_loader, desc="Collecting NON_MEMBER probabilities"):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            top_p, _ = probs.topk(10, dim=1)
            non_member_probs.append(top_p.cpu().numpy())

    member_probs = np.concatenate(member_probs, axis=0)
    non_member_probs = np.concatenate(non_member_probs, axis=0)

    return member_probs, non_member_probs

def create_shadow_attack_dataset(shadow_model, train_loader, test_loader, device, output_dim=10):
    """
    Builds a DataFrame that merges 'member' (train_loader) and 'non-member' (test_loader)
    samples. Each row is: [top_0_prob, top_1_prob, ..., top_{output_dim-1}_prob, is_member]

    :param shadow_model: One shadow model (trained on its subset)
    :param train_loader: DataLoader that yields shadow subset used during training (members).
    :param test_loader: DataLoader that yields shadow subset not used in training (non-members).
    :param device: 'cpu' or 'cuda'
    :param output_dim: how many top probabilities to include (default 10 for CIFAR-10)
    :return: A DataFrame with shape: (#samples_member + #samples_nonmember, output_dim+1)
    """
    columns = [f"top_{i}_prob" for i in range(output_dim)]
    member_probs, nonmember_probs = _make_member_nonmember_dataset(
        shadow_model, train_loader, test_loader, device
    )
    df_member = pd.DataFrame(member_probs, columns=columns)
    df_member["is_member"] = 1

    df_nonmember = pd.DataFrame(nonmember_probs, columns=columns)
    df_nonmember["is_member"] = 0

    df_attack = pd.concat([df_member, df_nonmember], ignore_index=True)
    return df_attack

def train_attack_model(df_attack, CFG):    
    """
    Train the final membership inference classifier (CatBoost) on 'df_attack',
    which has columns [top_0_prob, ..., top_{k-1}_prob, is_member].
    Returns the trained CatBoost model. Also saves the ROC curve and model checkpoint.

    :param df_attack: DataFrame of shape [N, k+1] with is_member label.
    :param config: Global config dict with attack and path info
    :return: trained CatBoostClassifier
    """
    # 1) Split data
    y = df_attack["is_member"]
    x = df_attack.drop(["is_member"], axis=1)

    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, random_state=42
    # )

    # 2) Build CatBoost model from config
    model = CatBoostClassifier(
        iterations=CFG.attack.catboost.iterations,
        depth=CFG.attack.catboost.depth,
        learning_rate=CFG.attack.catboost.learning_rate,
        loss_function=CFG.attack.catboost.loss_function,
        verbose=True
    )

    # 3) train
    model.fit(x, y)
    
    # 4) Save attack model
    os.makedirs(CFG.paths.attack_dir, exist_ok=True)
    attack_model_path = os.path.join(
        CFG.paths.attack_dir,
        f"catboost_attack_{model.__class__.__name__}"
    )
    model.save_model(attack_model_path)
    print(f"[Attack Model] Saved to {attack_model_path}")

    return model


def _train_shadow_model(model, train_loader, eval_loader, config, save_path, device):
    """
    Training function for shadow models with improved training loop.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        eval_loader: Evaluation data loader
        config: Configuration object
        save_path: Path to save model
        device: Device to train on
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    epochs = config.training.epochs
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Evaluation phase
        model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                eval_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        eval_acc = 100 * correct / total
        eval_loss = eval_loss / len(eval_loader)
        
        # Update learning rate
        scheduler.step(eval_acc)
        
        # Save best model
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved! Accuracy: {eval_acc:.2f}%")
        
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch [{epoch+1}/{epochs}]")
            print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"    Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%")


def _create_simple_model(num_classes: int = 10) -> nn.Module:
    """Create a CNN model for shadow models."""
    return nn.Sequential(
        # First conv block
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        
        # Second conv block
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        
        # Third conv block
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        
        # Fully connected layers
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )