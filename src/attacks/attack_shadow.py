# logic for training shadow models

import numpy as np
import os
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from src.new_utils.data import create_subset_dataloader
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from typing import List, Tuple, Dict

from src.new_utils.models import create_model


class ShadowAttack:
    """
    Shadow Model Attack for Membership Inference.
    
    This attack trains multiple shadow models on synthetic data to learn
    the behavior of models on member vs non-member data, then uses this
    knowledge to attack the target model.
    """
    
    def __init__(self, config, device: str = 'cpu'):
        """
        Initialize Shadow Attack.
        
        Args:
            config: Configuration object (dataclass-based)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.config = config
        self.device = device
        self.shadow_models = []
        self.attack_model = None
        self.name = "shadow"
        
    def train_shadow_models(self, full_dataset) -> List[Tuple]:
        """Train shadow models and return model info."""
        return train_shadow_models(self.config, full_dataset)
        
    def create_attack_dataset(self, shadow_models_info, full_dataset):
        """
        Create attack dataset from shadow models.
        """
        attack_datasets = []
        
        for i, (shadow_model, train_idx, eval_idx, test_idx) in enumerate(shadow_models_info):
            print(f"Creating attack dataset from shadow model {i}")
            
            # Fix data loader creation with class-based config
            train_loader = create_subset_dataloader(
                full_dataset, train_idx,
                batch_size=self.config.data.eval_batch_size,
                shuffle=False, num_workers=self.config.data.num_workers
            )
            test_loader = create_subset_dataloader(
                full_dataset, test_idx,
                batch_size=self.config.data.eval_batch_size,
                shuffle=False, num_workers=self.config.data.num_workers
            )
            
            df_attack = create_shadow_attack_dataset(
                shadow_model, train_loader, test_loader, self.device
            )
            attack_datasets.append(df_attack)
            
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

    def run(self, target_model, train_dataset, test_dataset, device):
        """
        Run the shadow attack and return results dict.
        """
        self.device = device
        
        # Train shadow models
        print("Training shadow models...")
        shadow_models_info = self.train_shadow_models(train_dataset)
        
        # Create attack dataset from shadow models
        print("Creating attack dataset...")
        df_attack = self.create_attack_dataset(shadow_models_info, train_dataset)
        
        # Train attack model
        print("Training attack classifier...")
        self.train_attack_model(df_attack)

        # Create data loaders for evaluation
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
        
        # Run attack on both train and test sets
        print("Running attack on training data...")
        train_results = self.infer_membership(target_model, train_loader)
        
        print("Running attack on test data...")
        test_results = self.infer_membership(target_model, test_loader)
        
        # Create ground truth labels
        train_labels = np.ones(len(train_results['predictions']))  # Training data = members
        test_labels = np.zeros(len(test_results['predictions']))   # Test data = non-members
        
        # Combine results
        all_predictions = np.concatenate([train_results['predictions'], test_results['predictions']])
        all_probabilities = np.concatenate([train_results['probabilities'], test_results['probabilities']])
        all_true_labels = np.concatenate([train_labels, test_labels])
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(all_true_labels, all_predictions)
        try:
            auc = roc_auc_score(all_true_labels, all_probabilities)
        except:
            auc = 0.5
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Calculate attack advantage
        train_accuracy = accuracy_score(train_labels, train_results['predictions'])
        test_accuracy = accuracy_score(test_labels, test_results['predictions'])
        attack_advantage = train_accuracy + test_accuracy - 1
        
        results = {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'attack_advantage': attack_advantage,
            'all_predictions': all_predictions,
            'all_true_labels': all_true_labels,
            'all_probabilities': all_probabilities
        }
        
        print(f"Shadow Attack Results: AUC = {auc:.4f}, Accuracy = {accuracy:.4f}, Advantage = {attack_advantage:.4f}")
        
        return results

def train_shadow_models(config, full_dataset):
    """
    Trains (or loads) multiple shadow models.
    """
    num_shadow = config.attack.shadow.num_shadows
    shadow_train_size = config.attack.shadow.shadow_train_size
    shadow_eval_size = config.attack.shadow.shadow_eval_size
    shadow_test_size = shadow_eval_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories for saving model checkpoints and indices
    model_save_dir = os.path.join(config.output.save_dir, "shadow_models")
    os.makedirs(model_save_dir, exist_ok=True)

    shadow_indices_save_dir = model_save_dir
    os.makedirs(shadow_indices_save_dir, exist_ok=True)

    indices = np.arange(len(full_dataset))
    shadow_models_info = []
    for i in range(num_shadow):
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

        train_loader = create_subset_dataloader(
            full_dataset,
            train_idx,
            batch_size=config.data.train_batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )
        eval_loader = create_subset_dataloader(
            full_dataset,
            eval_idx,
            batch_size=config.data.eval_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )

        # 3) Create or load the shadow model
        save_path = os.path.join(model_save_dir, f"shadow_model_{i}.pth")        
        
        # Create model using simple model creation
        shadow_model = create_model(config.model.num_classes).to(device)

        if os.path.exists(save_path):
            print(f"[Shadow {i}] Found checkpoint at {save_path}. Loading...")
            try:
                shadow_model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            except TypeError:
                shadow_model.load_state_dict(torch.load(save_path, map_location=device))
        else:
            print(f"[Shadow {i}] No checkpoint at {save_path}. Training new shadow model...")
            _train_shadow_model(shadow_model, train_loader, eval_loader, config, save_path, device)

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

# Remove duplicate function definition - keeping only the original one above


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
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
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

def train_attack_model(df_attack, config):
    """
    Train the attack model on shadow model data.
    
    Args:
        df_attack: DataFrame with shadow model data (features and membership labels)
        config: Configuration object
        
    Returns:
        Trained attack model
    """
    print("Training attack model on shadow data...")
    
    # Prepare features and target
    X = df_attack.drop("is_member", axis=1)
    y = df_attack["is_member"]
    
    # Use CatBoost as the attack model (good performance with default parameters)
    attack_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        verbose=50,
        random_seed=42
    )
    
    # Train the model
    attack_model.fit(X, y)
    
    # Print model accuracy
    train_preds = attack_model.predict(X)
    train_acc = accuracy_score(y, train_preds)
    print(f"Attack model training accuracy: {train_acc:.4f}")
    
    return attack_model