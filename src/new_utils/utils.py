# utility functions (early stopping classes, metics, etc.)

from pathlib import Path
import random
import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
import seaborn as sns



def compute_accuracy(model, data_loader, device):
    """Compute accuracy of a model on a given dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return 100 * correct / total


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

class EarlyStopPatience:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_metric = None
    
    def __call__(self, current_metric):
        if self.best_metric is None:
            self.best_metric = current_metric
            return False

        if current_metric <= self.best_metric:
            self.counter += 1
        else:
            self.best_metric = current_metric
            self.counter = 0
        
        if self.counter >= self.patience:
            return True
        return False
    


def evaluate_membership_inference(
    X_features: pd.DataFrame,
    y_true: np.ndarray,
    attack_model,
    config: dict,
    prefix: str = "attack_S"
):
    """
    Evaluate membership inference results on a given set of features/labels.

    :param X_features: DataFrame of shape [N, top_k], e.g. top_{i}_prob columns
    :param y_true: array-like of shape [N], membership ground-truth (0 or 1)
    :param attack_model: a trained classifier (CatBoost, RandomForest, etc.) 
                        that supports predict(...) and optionally predict_proba(...)
    :param config: your global config dict, containing e.g. paths["assets_dir"] for saving plots
    :param prefix: a string prefix for naming the output (e.g., "attack_S", "attack_R")
    :return: a dict of { "accuracy": float, "precision": float, "recall": float, "f1": float, "auc": float }
    """
    # 1) Predict membership labels
    y_pred = attack_model.predict(X_features)

    # 2) Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    # We'll compute AUC in two possible ways:
    # a) If the classifier has predict_proba => use that for a continuous score
    # b) Otherwise, we treat the predicted label as 0/1 (not recommended for a real ROC)
    try:
        y_scores = attack_model.predict_proba(X_features)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
    except AttributeError:
        # If no predict_proba => we do a 0/1 fallback (less informative)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
    
    roc_auc = auc(fpr, tpr)

    # 3) Print or log the metrics
    print(f"[{prefix}] Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")

    # 4) Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {prefix}")
    plt.legend(loc="lower right")
    plt.grid(True)

    assets_dir = config["paths"]["assets_dir"]
    os.makedirs(assets_dir, exist_ok=True)
    save_path = os.path.join(assets_dir, f"roc_{prefix}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[{prefix}] ROC curve saved => {save_path}")

    # 5) Return metrics in a dict
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": roc_auc
    }
    

# def plot_training_metrics(history, save_dir, model_name):
#     """Plot and save training metrics."""
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
    
#     # Set style
#     plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
#     sns.set_palette("husl")
    
#     # Create figure with subplots
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#     fig.suptitle(f'Training Metrics - {model_name}', fontsize=16)
    
#     # Plot training and validation loss
#     if 'train_loss' in history and 'val_loss' in history:
#         axes[0, 0].plot(history['train_loss'], label='Training Loss', marker='o')
#         axes[0, 0].plot(history['val_loss'], label='Validation Loss', marker='o')
#         axes[0, 0].set_xlabel('Epoch')
#         axes[0, 0].set_ylabel('Loss')
#         axes[0, 0].set_title('Training and Validation Loss')
#         axes[0, 0].legend()
#         axes[0, 0].grid(True)
    
#     # Plot training and validation accuracy
#     if 'train_acc' in history and 'val_acc' in history:
#         axes[0, 1].plot(history['train_acc'], label='Training Accuracy', marker='o')
#         axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', marker='o')
#         axes[0, 1].set_xlabel('Epoch')
#         axes[0, 1].set_ylabel('Accuracy')
#         axes[0, 1].set_title('Training and Validation Accuracy')
#         axes[0, 1].legend()
#         axes[0, 1].grid(True)
    
#     # Plot learning rate if available
#     if 'lr' in history:
#         axes[1, 0].plot(history['lr'], label='Learning Rate', marker='o')
#         axes[1, 0].set_xlabel('Epoch')
#         axes[1, 0].set_ylabel('Learning Rate')
#         axes[1, 0].set_title('Learning Rate Schedule')
#         axes[1, 0].legend()
#         axes[1, 0].grid(True)
    
#     # Plot privacy budget if available
#     if 'epsilon' in history:
#         axes[1, 1].plot(history['epsilon'], label='Privacy Budget (ε)', marker='o')
#         axes[1, 1].set_xlabel('Epoch')
#         axes[1, 1].set_ylabel('Epsilon')
#         axes[1, 1].set_title('Privacy Budget Over Time')
#         axes[1, 1].legend()
#         axes[1, 1].grid(True)
    
#     plt.tight_layout()
#     plt.savefig(save_dir / f'training_metrics_{model_name}.png', dpi=300, bbox_inches='tight')
#     plt.close()
