# utility functions (early stopping classes, metics, etc.)

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