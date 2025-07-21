from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


def evaluate_attack(
    predictions,
    labels,
    save_dir=None,
    attack_name="attack"
):
    """Evaluate attack with standard metrics and visualization"""

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics
    roc_auc = auc(fpr, tpr)
    y_pred = (predictions >= optimal_threshold).astype(int)
    cm = confusion_matrix(labels, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate accuracy measures
    optimal_accuracy = (tp + tn) / (tp + tn + fp + fn)
    default_accuracy = np.mean((predictions >= 0.5).astype(int) == labels)

    # Calculate attack advantage
    tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
    attack_advantage = tpr_value - fpr_value
    
    # Combine results
    results = {
        'auc': float(roc_auc),
        'accuracy_optimal': float(optimal_accuracy),
        'accuracy_default': float(default_accuracy),
        'optimal_threshold': float(optimal_threshold),
        'confusion_matrix': cm.tolist(),
        'attack_advantage': float(attack_advantage),
        'member_prediction_mean': float(np.mean(predictions[labels == 1])),
        'non_member_prediction_mean': float(np.mean(predictions[labels == 0])),
    }
    
    # Create visualization if save_dir is not None:
    if save_dir:
        visualize_attack_performance(
            predictions, labels, results,
            Path(save_dir) / f"{attack_name}_performance.png"
        )
    return results

def visualize_attack_performance(predictions, labels, results, save_path):
    """Create attack performance charts"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Attack Performance for " + save_path.stem, fontsize=14)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, predictions)
    axes[0, 0].plot(fpr, tpr, label=f"ROC (AUC = {results['auc']:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Guess')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    
    # Prediction distributions
    axes[0, 1].hist(predictions[labels == 0], bins=20, alpha=0.5, label='Non-Members', color='blue')
    axes[0, 1].hist(predictions[labels == 1], bins=20, alpha=0.5, label='Members', color='orange')
    axes[0, 1].axvline(results['optimal_threshold'], color='red', linestyle='--', label=f'Optimal Threshold: {results['optimal_threshold']:.3f}')
    axes[0, 1].set_xlabel('Attack Output (Membership Score)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Prediction Distributions')
    axes[0, 1].legend()
    
    # Confusion Matrix
    cm = results['confusion_matrix']
    axes[1, 0].matshow(cm, cmap='Blues', alpha=0.7)
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, str(cm[i][j]), va='center', ha='center', size=12)
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(['Non-Member', 'Member'])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(['Non-Member', 'Member'])
    
    # Key metrics
    metrics = [
        f"AUC: {results['auc']:.4f}",
        f"Accuracy: {results['accuracy_optimal']:.4f}",
        f"Attack Advantage: {results['attack_advantage']:.4f}",
        f"Member Mean: {results['member_prediction_mean']:.4f}",
        f"Non-member Mean: {results['non_member_prediction_mean']:.4f}"
    ]
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.5, '\n'.join(metrics), 
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()