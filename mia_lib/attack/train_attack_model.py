# logic to train final MIA (e.g. XGBoost, LightGBM, CatBoost, etc.) model

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

def train_attack_model(df_attack, config):
    """
    Train a membership inference classifier on top of the 'attack dataset'.
    """
    y = df_attack["is_member"]
    x = df_attack.drop(["is_member"], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    cat_config = config["attack"]["catboost"]
    model = CatBoostClassifier(
        iterations=cat_config["iterations"],
        depth=cat_config["depth"],
        learning_rate=cat_config["learning_rate"],
        loss_function=cat_config["loss_function"],
        verbose=True
    )

    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    print(f"[Attack Model] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")


    # Plot ROC curve
    y_pred_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    print(f"[Attack Model] AUC: {roc_auc:.4f}")

    # Save figure
    assets_dir = config["paths"]["assets_dir"]
    os.makedirs(assets_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"MIA ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Membership Inference Attack - ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(assets_dir, f"roc_curve_{model.__class__.__name__}_{accuracy:.4f}.png"))
    plt.close()
    print(f"[Attack Model] ROC curve saved to {assets_dir}")

    # Save model
    attack_model_path = os.path.join(
        config["paths"]["attack_save_dir"],
        f"catboost_attack_{model.__class__.__name__}_{accuracy:.4f}"
    )

    model.save_model(attack_model_path)
    print(f"[Attack Model] Saved to {attack_model_path}")

    return model