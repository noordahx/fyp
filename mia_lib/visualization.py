"""
Comprehensive visualization module for MIA and DP results.

This module provides functions to generate detailed visual reports including:
- Training curves and overfitting analysis
- MIA attack performance metrics
- Privacy-utility tradeoff curves
- Score distributions and ROC curves
- Summary dashboards
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

class MIAVisualizer:
    """Visualizer for Membership Inference Attack results."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_training_report(self, training_history, model_info, save_name="training_report"):
        """Create comprehensive training analysis report."""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Training and validation curves
        ax1 = plt.subplot(2, 3, 1)
        if 'train_accuracies' in training_history and 'test_accuracies' in training_history:
            epochs = range(1, len(training_history['train_accuracies']) + 1)
            plt.plot(epochs, training_history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
            plt.plot(epochs, training_history['test_accuracies'], 'r-', label='Test Accuracy', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training vs Test Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Highlight overfitting region
            final_train = training_history['train_accuracies'][-1]
            final_test = training_history['test_accuracies'][-1]
            gap = final_train - final_test
            plt.axhline(y=final_train, color='b', linestyle='--', alpha=0.5)
            plt.axhline(y=final_test, color='r', linestyle='--', alpha=0.5)
            plt.fill_between(epochs, final_test, final_train, alpha=0.2, color='orange', 
                           label=f'Overfitting Gap: {gap:.1f}%')
        
        ax2 = plt.subplot(2, 3, 2)
        if 'train_losses' in training_history and 'test_losses' in training_history:
            epochs = range(1, len(training_history['train_losses']) + 1)
            plt.plot(epochs, training_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
            plt.plot(epochs, training_history['test_losses'], 'r-', label='Test Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training vs Test Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
        
        # Overfitting progression
        ax3 = plt.subplot(2, 3, 3)
        if 'train_accuracies' in training_history and 'test_accuracies' in training_history:
            epochs = range(1, len(training_history['train_accuracies']) + 1)
            gaps = [t - v for t, v in zip(training_history['train_accuracies'], training_history['test_accuracies'])]
            plt.plot(epochs, gaps, 'g-', linewidth=2, marker='o', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('Overfitting Gap (%)')
            plt.title('Overfitting Progression')
            plt.grid(True, alpha=0.3)
            
            # Color code regions
            plt.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Excellent (>30%)')
            plt.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Good (>15%)')
            plt.axhline(y=5, color='yellow', linestyle='--', alpha=0.7, label='Moderate (>5%)')
            plt.legend()
        
        # Model information
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        info_text = f"""
Model Information:
• Dataset: {model_info.get('dataset', 'N/A')}
• Model Size: {model_info.get('model_size', 'N/A')}
• Parameters: {model_info.get('total_params', 'N/A'):,}
• Training Samples: {model_info.get('train_size', 'N/A'):,}
• Test Samples: {model_info.get('test_size', 'N/A'):,}
• Epochs: {model_info.get('epochs', 'N/A')}
• Batch Size: {model_info.get('batch_size', 'N/A')}
• Learning Rate: {model_info.get('lr', 'N/A')}
• Weight Decay: {model_info.get('weight_decay', 'N/A')}

Final Results:
• Train Accuracy: {training_history.get('train_accuracies', [0])[-1]:.2f}%
• Test Accuracy: {training_history.get('test_accuracies', [0])[-1]:.2f}%
• Overfitting Gap: {training_history.get('train_accuracies', [0])[-1] - training_history.get('test_accuracies', [0])[-1]:.2f}%
        """
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Training efficiency
        ax5 = plt.subplot(2, 3, 5)
        if 'train_accuracies' in training_history:
            epochs = range(1, len(training_history['train_accuracies']) + 1)
            train_acc = training_history['train_accuracies']
            
            # Calculate convergence metrics
            convergence_epoch = next((i for i, acc in enumerate(train_acc) if acc > 90), len(train_acc))
            
            plt.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
            plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% Threshold')
            plt.axvline(x=convergence_epoch+1, color='green', linestyle='--', alpha=0.7, 
                       label=f'Convergence: Epoch {convergence_epoch+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Training Accuracy (%)')
            plt.title('Training Convergence Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Overfitting risk assessment
        ax6 = plt.subplot(2, 3, 6)
        if 'train_accuracies' in training_history and 'test_accuracies' in training_history:
            final_gap = training_history['train_accuracies'][-1] - training_history['test_accuracies'][-1]
            
            # Risk categories
            categories = ['Low\n(<10%)', 'Moderate\n(10-20%)', 'High\n(20-35%)', 'Extreme\n(>35%)']
            colors = ['green', 'yellow', 'orange', 'red']
            values = [10, 20, 35, 50]
            
            bars = plt.bar(categories, values, color=colors, alpha=0.6)
            
            # Highlight current gap
            if final_gap < 10:
                highlight_idx = 0
            elif final_gap < 20:
                highlight_idx = 1
            elif final_gap < 35:
                highlight_idx = 2
            else:
                highlight_idx = 3
                
            bars[highlight_idx].set_alpha(1.0)
            bars[highlight_idx].set_edgecolor('black')
            bars[highlight_idx].set_linewidth(3)
            
            plt.axhline(y=final_gap, color='black', linestyle='-', linewidth=2, 
                       label=f'Current Gap: {final_gap:.1f}%')
            plt.ylabel('Overfitting Gap (%)')
            plt.title('Overfitting Risk Assessment')
            plt.legend()
        
        plt.suptitle(f'Training Analysis Report - {model_info.get("model_name", "Model")}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save report
        report_path = self.output_dir / f'{save_name}.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return report_path
    
    def create_mia_report(self, mia_results, model_info, save_name="mia_report"):
        """Create comprehensive MIA attack analysis report."""
        
        fig = plt.figure(figsize=(20, 15))
        
        # Attack performance comparison
        ax1 = plt.subplot(3, 4, 1)
        attacks = list(mia_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Create comparison data
        comparison_data = []
        for attack in attacks:
            for metric in metrics:
                if metric in mia_results[attack]:
                    comparison_data.append({
                        'Attack': attack.capitalize(),
                        'Metric': metric.capitalize(),
                        'Value': mia_results[attack][metric]
                    })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            pivot_df = df.pivot(index='Attack', columns='Metric', values='Value')
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                       cbar_kws={'label': 'Score'}, ax=ax1)
            ax1.set_title('Attack Performance Heatmap')
        
        # ROC Curves
        ax2 = plt.subplot(3, 4, 2)
        roc_curves_plotted = 0
        
        for attack_name, results in mia_results.items():
            if 'probabilities' in results and 'true_labels' in results:
                try:
                    probabilities = np.array(results['probabilities'])
                    true_labels = np.array(results['true_labels'])
                    
                    # Check if we have valid data
                    if len(probabilities) > 0 and len(true_labels) > 0 and len(probabilities) == len(true_labels):
                        # Check if we have both classes
                        if len(np.unique(true_labels)) > 1:
                            fpr, tpr, _ = roc_curve(true_labels, probabilities)
                            auc_score = results.get('auc', 0.5)
                            ax2.plot(fpr, tpr, linewidth=2, label=f'{attack_name.capitalize()} (AUC={auc_score:.3f})')
                            roc_curves_plotted += 1
                except Exception as e:
                    print(f"Warning: Could not plot ROC curve for {attack_name}: {e}")
                    continue
        
        # Always plot the random baseline
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
        
        # If no curves were plotted, add a message
        if roc_curves_plotted == 0:
            ax2.text(0.5, 0.5, 'No ROC curves available\n(Missing probability data)', 
                    ha='center', va='center', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # AUC Comparison
        ax3 = plt.subplot(3, 4, 3)
        attack_names = []
        auc_scores = []
        colors = []
        
        for attack_name, results in mia_results.items():
            if 'auc' in results:
                attack_names.append(attack_name.capitalize())
                auc_score = results['auc']
                auc_scores.append(auc_score)
                
                # Color code by performance
                if auc_score > 0.8:
                    colors.append('red')
                elif auc_score > 0.7:
                    colors.append('orange')
                elif auc_score > 0.6:
                    colors.append('yellow')
                else:
                    colors.append('green')
        
        bars = ax3.bar(attack_names, auc_scores, color=colors, alpha=0.7)
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax3.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (0.7)')
        ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Excellent (0.8)')
        ax3.set_ylabel('AUC Score')
        ax3.set_title('Attack AUC Comparison')
        ax3.legend()
        ax3.set_ylim(0.4, 1.0)
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Score distributions for each attack
        distribution_plots = 0
        for i, (attack_name, results) in enumerate(mia_results.items()):
            if distribution_plots >= 3:  # Limit to 3 attacks for space
                break
                
            if 'probabilities' in results and 'true_labels' in results:
                try:
                    ax = plt.subplot(3, 4, 5 + distribution_plots)
                    
                    probabilities = np.array(results['probabilities'])
                    true_labels = np.array(results['true_labels'])
                    
                    # Check if we have valid data
                    if len(probabilities) > 0 and len(true_labels) > 0 and len(probabilities) == len(true_labels):
                        member_scores = probabilities[true_labels == 1]
                        non_member_scores = probabilities[true_labels == 0]
                        
                        if len(member_scores) > 0 and len(non_member_scores) > 0:
                            # Handle constant values
                            if np.std(member_scores) > 1e-6:
                                sns.kdeplot(member_scores, label='Members', fill=True, alpha=0.6, ax=ax, color='blue')
                            else:
                                ax.axvline(member_scores[0], color='blue', label='Members', linewidth=3)
                            
                            if np.std(non_member_scores) > 1e-6:
                                sns.kdeplot(non_member_scores, label='Non-members', fill=True, alpha=0.6, ax=ax, color='red')
                            else:
                                ax.axvline(non_member_scores[0], color='red', label='Non-members', linewidth=3)
                            
                            ax.set_xlabel('Attack Score')
                            ax.set_ylabel('Density')
                            ax.set_title(f'{attack_name.capitalize()} Score Distribution')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            distribution_plots += 1
                        else:
                            # No valid scores for one or both classes
                            ax.text(0.5, 0.5, f'No score distribution available\nfor {attack_name.capitalize()}', 
                                   ha='center', va='center', transform=ax.transAxes,
                                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                            ax.set_title(f'{attack_name.capitalize()} Score Distribution')
                            distribution_plots += 1
                    else:
                        # Invalid data
                        ax.text(0.5, 0.5, f'Invalid data\nfor {attack_name.capitalize()}', 
                               ha='center', va='center', transform=ax.transAxes,
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                        ax.set_title(f'{attack_name.capitalize()} Score Distribution')
                        distribution_plots += 1
                        
                except Exception as e:
                    print(f"Warning: Could not plot score distribution for {attack_name}: {e}")
                    continue
        
        # Attack summary
        ax_summary = plt.subplot(3, 4, 9)
        ax_summary.axis('off')
        
        summary_text = "Attack Performance Summary:\n\n"
        for attack_name, results in mia_results.items():
            auc = results.get('auc', 0.5)
            acc = results.get('accuracy', 0.5)
            
            if auc > 0.8:
                status = "🟢 Excellent"
            elif auc > 0.7:
                status = "🟡 Good"
            elif auc > 0.6:
                status = "🟠 Moderate"
            else:
                status = "🔴 Poor"
            
            summary_text += f"{attack_name.capitalize()}:\n"
            summary_text += f"  • AUC: {auc:.3f} {status}\n"
            summary_text += f"  • Accuracy: {acc:.3f}\n\n"
        
        # Add vulnerability assessment
        best_auc = max([results.get('auc', 0.5) for results in mia_results.values()])
        summary_text += f"Overall Vulnerability: "
        if best_auc > 0.8:
            summary_text += "🔴 HIGH RISK"
        elif best_auc > 0.7:
            summary_text += "🟠 MODERATE RISK"
        elif best_auc > 0.6:
            summary_text += "🟡 LOW RISK"
        else:
            summary_text += "🟢 MINIMAL RISK"
        
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(f'MIA Attack Analysis Report - {model_info.get("model_name", "Model")}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save report
        report_path = self.output_dir / f'{save_name}.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return report_path
    
    def create_dp_report(self, dp_results, save_name="dp_report"):
        """Create differential privacy analysis report."""
        
        if not dp_results or len(dp_results) < 2:
            print("Insufficient DP results for comparison report")
            return None
        
        fig = plt.figure(figsize=(20, 12))
        
        # Extract epsilon values and metrics
        epsilons = []
        model_accs = []
        attack_accs = []
        attack_aucs = []
        
        for eps_str, results in dp_results.items():
            try:
                eps = float(eps_str) if eps_str != 'standard' else float('inf')
                epsilons.append(eps)
                model_accs.append(results.get('model_accuracy', 0))
                attack_accs.append(results.get('attack_accuracy', 0.5))
                attack_aucs.append(results.get('attack_auc', 0.5))
            except:
                continue
        
        # Sort by epsilon
        sorted_data = sorted(zip(epsilons, model_accs, attack_accs, attack_aucs))
        epsilons, model_accs, attack_accs, attack_aucs = zip(*sorted_data)
        
        # Privacy-Utility Tradeoff
        ax1 = plt.subplot(2, 3, 1)
        ax1.semilogx(epsilons, model_accs, 'bo-', linewidth=2, markersize=8, label='Model Accuracy')
        ax1.set_xlabel('Privacy Budget (ε)')
        ax1.set_ylabel('Model Accuracy')
        ax1.set_title('Privacy-Utility Tradeoff')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Attack Success vs Privacy
        ax2 = plt.subplot(2, 3, 2)
        ax2.semilogx(epsilons, attack_accs, 'ro-', linewidth=2, markersize=8, label='Attack Accuracy')
        ax2.semilogx(epsilons, attack_aucs, 'go-', linewidth=2, markersize=8, label='Attack AUC')
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
        ax2.set_xlabel('Privacy Budget (ε)')
        ax2.set_ylabel('Attack Performance')
        ax2.set_title('Attack Performance vs Privacy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Privacy Efficiency
        ax3 = plt.subplot(2, 3, 3)
        privacy_efficiency = [acc / (auc if auc > 0.5 else 0.5) for acc, auc in zip(model_accs, attack_aucs)]
        ax3.semilogx(epsilons, privacy_efficiency, 'mo-', linewidth=2, markersize=8)
        ax3.set_xlabel('Privacy Budget (ε)')
        ax3.set_ylabel('Privacy Efficiency (Utility/Vulnerability)')
        ax3.set_title('Privacy Efficiency Analysis')
        ax3.grid(True, alpha=0.3)
        
        # Comparison table
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        
        table_data = []
        for eps, model_acc, attack_acc, attack_auc in zip(epsilons, model_accs, attack_accs, attack_aucs):
            eps_str = f"{eps:.1f}" if eps != float('inf') else "∞"
            table_data.append([eps_str, f"{model_acc:.3f}", f"{attack_acc:.3f}", f"{attack_auc:.3f}"])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['ε', 'Model Acc', 'Attack Acc', 'Attack AUC'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Privacy Budget Comparison')
        
        # Recommendations
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        # Find optimal epsilon
        best_efficiency_idx = np.argmax(privacy_efficiency)
        optimal_eps = epsilons[best_efficiency_idx]
        
        recommendations = f"""
Privacy Analysis Recommendations:

Optimal Privacy Budget: ε = {optimal_eps:.1f}
• Model Accuracy: {model_accs[best_efficiency_idx]:.3f}
• Attack AUC: {attack_aucs[best_efficiency_idx]:.3f}
• Privacy Efficiency: {privacy_efficiency[best_efficiency_idx]:.2f}

Privacy Levels:
• High Privacy (ε < 1.0): Strong protection, lower utility
• Moderate Privacy (1.0 ≤ ε < 5.0): Balanced approach
• Low Privacy (ε ≥ 5.0): Higher utility, weaker protection

Current Status:
• Best Model Performance: {max(model_accs):.3f}
• Lowest Attack Success: {min(attack_aucs):.3f}
• Privacy-Utility Sweet Spot: ε = {optimal_eps:.1f}
        """
        
        ax5.text(0.05, 0.95, recommendations, transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Privacy risk assessment
        ax6 = plt.subplot(2, 3, 6)
        risk_levels = ['Low Risk\n(AUC < 0.6)', 'Moderate Risk\n(0.6 ≤ AUC < 0.7)', 
                      'High Risk\n(0.7 ≤ AUC < 0.8)', 'Critical Risk\n(AUC ≥ 0.8)']
        risk_colors = ['green', 'yellow', 'orange', 'red']
        
        # Count models in each risk category
        risk_counts = [0, 0, 0, 0]
        for auc in attack_aucs:
            if auc < 0.6:
                risk_counts[0] += 1
            elif auc < 0.7:
                risk_counts[1] += 1
            elif auc < 0.8:
                risk_counts[2] += 1
            else:
                risk_counts[3] += 1
        
        bars = ax6.bar(risk_levels, risk_counts, color=risk_colors, alpha=0.7)
        ax6.set_ylabel('Number of Models')
        ax6.set_title('Privacy Risk Distribution')
        
        # Add count labels
        for bar, count in zip(bars, risk_counts):
            if count > 0:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Differential Privacy Analysis Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save report
        report_path = self.output_dir / f'{save_name}.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return report_path
    
    def create_summary_dashboard(self, training_history, mia_results, model_info, save_name="summary_dashboard"):
        """Create a comprehensive summary dashboard."""
        
        fig = plt.figure(figsize=(24, 16))
        
        # Title and metadata
        fig.suptitle(f'ML Privacy Analysis Dashboard - {model_info.get("model_name", "Model")}', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # Model overview (top left)
        ax1 = plt.subplot(3, 4, 1)
        ax1.axis('off')
        
        # Calculate key metrics
        final_train_acc = training_history.get('train_accuracies', [0])[-1]
        final_test_acc = training_history.get('test_accuracies', [0])[-1]
        overfitting_gap = final_train_acc - final_test_acc
        best_auc = max([results.get('auc', 0.5) for results in mia_results.values()]) if mia_results else 0.5
        
        # Risk assessment
        if overfitting_gap > 30 and best_auc > 0.8:
            risk_level = "🔴 CRITICAL"
            risk_color = "red"
        elif overfitting_gap > 20 and best_auc > 0.7:
            risk_level = "🟠 HIGH"
            risk_color = "orange"
        elif overfitting_gap > 10 and best_auc > 0.6:
            risk_level = "🟡 MODERATE"
            risk_color = "yellow"
        else:
            risk_level = "🟢 LOW"
            risk_color = "green"
        
        overview_text = f"""
MODEL OVERVIEW
{'='*30}

Dataset: {model_info.get('dataset', 'N/A').upper()}
Architecture: {model_info.get('model_size', 'N/A').upper()}
Parameters: {model_info.get('total_params', 0):,}

PERFORMANCE METRICS
{'='*30}

Training Accuracy: {final_train_acc:.1f}%
Test Accuracy: {final_test_acc:.1f}%
Overfitting Gap: {overfitting_gap:.1f}%

PRIVACY VULNERABILITY
{'='*30}

Best Attack AUC: {best_auc:.3f}
Privacy Risk: {risk_level}

Training Time: {model_info.get('training_time', 'N/A')}
Device: {model_info.get('device', 'N/A').upper()}
        """
        
        ax1.text(0.05, 0.95, overview_text, transform=ax1.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=risk_color, alpha=0.3))
        
        # Training curves (top middle)
        ax2 = plt.subplot(3, 4, 2)
        if 'train_accuracies' in training_history:
            epochs = range(1, len(training_history['train_accuracies']) + 1)
            ax2.plot(epochs, training_history['train_accuracies'], 'b-', linewidth=3, label='Training')
            ax2.plot(epochs, training_history['test_accuracies'], 'r-', linewidth=3, label='Test')
            ax2.fill_between(epochs, training_history['test_accuracies'], 
                           training_history['train_accuracies'], alpha=0.3, color='orange')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training Progress')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Attack performance radar chart (top right)
        ax3 = plt.subplot(3, 4, 3, projection='polar')
        if mia_results:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for attack_name, results in mia_results.items():
                values = [results.get(metric, 0) for metric in metrics]
                values += values[:1]  # Complete the circle
                
                ax3.plot(angles, values, 'o-', linewidth=2, label=attack_name.capitalize())
                ax3.fill(angles, values, alpha=0.25)
            
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels([m.capitalize() for m in metrics])
            ax3.set_ylim(0, 1)
            ax3.set_title('Attack Performance Radar')
            ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # ROC curves (middle left)
        ax4 = plt.subplot(3, 4, 5)
        roc_curves_plotted = 0
        
        for attack_name, results in mia_results.items():
            if 'probabilities' in results and 'true_labels' in results:
                try:
                    probabilities = np.array(results['probabilities'])
                    true_labels = np.array(results['true_labels'])
                    
                    # Check if we have valid data
                    if len(probabilities) > 0 and len(true_labels) > 0 and len(probabilities) == len(true_labels):
                        # Check if we have both classes
                        if len(np.unique(true_labels)) > 1:
                            fpr, tpr, _ = roc_curve(true_labels, probabilities)
                            auc_score = results.get('auc', 0.5)
                            ax4.plot(fpr, tpr, linewidth=3, label=f'{attack_name.capitalize()} (AUC={auc_score:.3f})')
                            roc_curves_plotted += 1
                except Exception as e:
                    print(f"Warning: Could not plot ROC curve for {attack_name}: {e}")
                    continue
        
        # Always plot the random baseline
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random (AUC=0.5)')
        
        # If no curves were plotted, add a message
        if roc_curves_plotted == 0:
            ax4.text(0.5, 0.5, 'No ROC curves available\n(Missing probability data)', 
                    ha='center', va='center', transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        
        # Score distributions (middle)
        distribution_plots = 0
        for i, (attack_name, results) in enumerate(mia_results.items()):
            if distribution_plots >= 2:  # Limit to 2 attacks for space
                break
                
            ax = plt.subplot(3, 4, 6 + distribution_plots)
            
            if 'probabilities' in results and 'true_labels' in results:
                try:
                    probabilities = np.array(results['probabilities'])
                    true_labels = np.array(results['true_labels'])
                    
                    # Check if we have valid data
                    if len(probabilities) > 0 and len(true_labels) > 0 and len(probabilities) == len(true_labels):
                        member_scores = probabilities[true_labels == 1]
                        non_member_scores = probabilities[true_labels == 0]
                        
                        if len(member_scores) > 0 and len(non_member_scores) > 0:
                            if np.std(member_scores) > 1e-6:
                                ax.hist(member_scores, bins=20, alpha=0.7, label='Members', color='blue', density=True)
                            else:
                                ax.axvline(member_scores[0], color='blue', label='Members', linewidth=3)
                                
                            if np.std(non_member_scores) > 1e-6:
                                ax.hist(non_member_scores, bins=20, alpha=0.7, label='Non-members', color='red', density=True)
                            else:
                                ax.axvline(non_member_scores[0], color='red', label='Non-members', linewidth=3)
                            
                            ax.set_xlabel('Attack Score')
                            ax.set_ylabel('Density')
                            ax.set_title(f'{attack_name.capitalize()} Scores')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            distribution_plots += 1
                        else:
                            # No valid scores
                            ax.text(0.5, 0.5, f'No score data\nfor {attack_name.capitalize()}', 
                                   ha='center', va='center', transform=ax.transAxes,
                                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                            ax.set_title(f'{attack_name.capitalize()} Scores')
                            distribution_plots += 1
                    else:
                        # Invalid data
                        ax.text(0.5, 0.5, f'Invalid data\nfor {attack_name.capitalize()}', 
                               ha='center', va='center', transform=ax.transAxes,
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                        ax.set_title(f'{attack_name.capitalize()} Scores')
                        distribution_plots += 1
                        
                except Exception as e:
                    print(f"Warning: Could not plot score distribution for {attack_name}: {e}")
                    ax.text(0.5, 0.5, f'Error plotting\n{attack_name.capitalize()}', 
                           ha='center', va='center', transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                    ax.set_title(f'{attack_name.capitalize()} Scores')
                    distribution_plots += 1
            else:
                # No probability data available
                ax.text(0.5, 0.5, f'No probability data\nfor {attack_name.capitalize()}', 
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_title(f'{attack_name.capitalize()} Scores')
                distribution_plots += 1
        
        # Attack comparison (bottom left)
        ax8 = plt.subplot(3, 4, 9)
        attack_names = []
        auc_scores = []
        colors = []
        
        for attack_name, results in mia_results.items():
            if 'auc' in results:
                attack_names.append(attack_name.capitalize())
                auc_score = results['auc']
                auc_scores.append(auc_score)
                
                if auc_score > 0.8:
                    colors.append('red')
                elif auc_score > 0.7:
                    colors.append('orange')
                elif auc_score > 0.6:
                    colors.append('yellow')
                else:
                    colors.append('green')
        
        bars = ax8.bar(attack_names, auc_scores, color=colors, alpha=0.8)
        ax8.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
        ax8.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Concerning')
        ax8.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Critical')
        ax8.set_ylabel('AUC Score')
        ax8.set_title('Attack Success Comparison')
        ax8.legend()
        ax8.set_ylim(0.4, 1.0)
        
        # Add value labels
        for bar, score in zip(bars, auc_scores):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Recommendations (bottom middle)
        ax9 = plt.subplot(3, 4, 10)
        ax9.axis('off')
        
        recommendations = f"""
RECOMMENDATIONS
{'='*25}

Model Security:
"""
        
        if best_auc > 0.8:
            recommendations += """
🔴 CRITICAL: High vulnerability
• Apply differential privacy
• Reduce model complexity
• Limit training data access
"""
        elif best_auc > 0.7:
            recommendations += """
🟠 HIGH: Moderate vulnerability
• Consider privacy techniques
• Monitor data access
• Regular security audits
"""
        elif best_auc > 0.6:
            recommendations += """
🟡 MODERATE: Some vulnerability
• Basic privacy measures
• Access controls
• Monitoring recommended
"""
        else:
            recommendations += """
🟢 LOW: Minimal vulnerability
• Standard security practices
• Regular monitoring
• Current approach acceptable
"""
        
        if overfitting_gap > 30:
            recommendations += """

Overfitting Mitigation:
• Reduce training epochs
• Add regularization
• Increase training data
• Use dropout/batch norm
"""
        
        ax9.text(0.05, 0.95, recommendations, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Privacy timeline (bottom right)
        ax10 = plt.subplot(3, 4, 11)
        if 'train_accuracies' in training_history:
            epochs = range(1, len(training_history['train_accuracies']) + 1)
            gaps = [t - v for t, v in zip(training_history['train_accuracies'], training_history['test_accuracies'])]
            
            # Color code by risk level
            colors = ['green' if gap < 15 else 'yellow' if gap < 25 else 'orange' if gap < 35 else 'red' for gap in gaps]
            ax10.scatter(epochs, gaps, c=colors, s=50, alpha=0.7)
            ax10.plot(epochs, gaps, 'k-', alpha=0.3)
            
            ax10.axhline(y=15, color='yellow', linestyle='--', alpha=0.5, label='Moderate Risk')
            ax10.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='High Risk')
            ax10.axhline(y=35, color='red', linestyle='--', alpha=0.5, label='Critical Risk')
            
            ax10.set_xlabel('Epoch')
            ax10.set_ylabel('Overfitting Gap (%)')
            ax10.set_title('Privacy Risk Timeline')
            ax10.legend()
            ax10.grid(True, alpha=0.3)
        
        # Footer with timestamp
        fig.text(0.5, 0.02, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.output_dir / f'{save_name}.png'
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return dashboard_path
    
    def generate_html_report(self, training_history, mia_results, model_info, dp_results=None, save_name="report"):
        """Generate comprehensive HTML report with all visualizations."""
        
        # Generate all visualizations
        training_report = self.create_training_report(training_history, model_info, f"{save_name}_training")
        mia_report = self.create_mia_report(mia_results, model_info, f"{save_name}_mia")
        dashboard = self.create_summary_dashboard(training_history, mia_results, model_info, f"{save_name}_dashboard")
        
        dp_report = None
        if dp_results:
            dp_report = self.create_dp_report(dp_results, f"{save_name}_dp")
        
        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Privacy Analysis Report - {model_info.get('model_name', 'Model')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .report-section {{
            margin: 30px 0;
            text-align: center;
        }}
        .report-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .metadata table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .metadata th, .metadata td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #bdc3c7;
        }}
        .metadata th {{
            background-color: #3498db;
            color: white;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            color: #7f8c8d;
        }}
        .risk-high {{ color: #e74c3c; font-weight: bold; }}
        .risk-medium {{ color: #f39c12; font-weight: bold; }}
        .risk-low {{ color: #27ae60; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 ML Privacy Analysis Report</h1>
        <p style="text-align: center; font-size: 18px; color: #7f8c8d;">
            Model: <strong>{model_info.get('model_name', 'Unknown')}</strong> | 
            Generated: <strong>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</strong>
        </p>
        
        <div class="metadata">
            <h3>📊 Model Information</h3>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Dataset</td><td>{model_info.get('dataset', 'N/A')}</td></tr>
                <tr><td>Architecture</td><td>{model_info.get('model_size', 'N/A')}</td></tr>
                <tr><td>Total Parameters</td><td>{model_info.get('total_params', 'N/A'):,}</td></tr>
                <tr><td>Training Samples</td><td>{model_info.get('train_size', 'N/A'):,}</td></tr>
                <tr><td>Test Samples</td><td>{model_info.get('test_size', 'N/A'):,}</td></tr>
                <tr><td>Training Epochs</td><td>{model_info.get('epochs', 'N/A')}</td></tr>
                <tr><td>Final Train Accuracy</td><td>{training_history.get('train_accuracies', [0])[-1]:.2f}%</td></tr>
                <tr><td>Final Test Accuracy</td><td>{training_history.get('test_accuracies', [0])[-1]:.2f}%</td></tr>
                <tr><td>Overfitting Gap</td><td>{training_history.get('train_accuracies', [0])[-1] - training_history.get('test_accuracies', [0])[-1]:.2f}%</td></tr>
            </table>
        </div>
        
        <div class="report-section">
            <h2>📈 Executive Summary Dashboard</h2>
            <img src="{dashboard.name}" alt="Summary Dashboard" class="report-image">
        </div>
        
        <div class="report-section">
            <h2>🎯 Training Analysis</h2>
            <img src="{training_report.name}" alt="Training Report" class="report-image">
        </div>
        
        <div class="report-section">
            <h2>🔍 Membership Inference Attack Analysis</h2>
            <img src="{mia_report.name}" alt="MIA Report" class="report-image">
            
            <div class="metadata">
                <h3>🎯 Attack Results Summary</h3>
                <table>
                    <tr><th>Attack Type</th><th>AUC Score</th><th>Accuracy</th><th>Risk Level</th></tr>
        """
        
        # Add attack results to table
        for attack_name, results in mia_results.items():
            auc = results.get('auc', 0.5)
            acc = results.get('accuracy', 0.5)
            
            if auc > 0.8:
                risk_class = "risk-high"
                risk_text = "🔴 CRITICAL"
            elif auc > 0.7:
                risk_class = "risk-high"
                risk_text = "🟠 HIGH"
            elif auc > 0.6:
                risk_class = "risk-medium"
                risk_text = "🟡 MODERATE"
            else:
                risk_class = "risk-low"
                risk_text = "🟢 LOW"
            
            html_content += f"""
                    <tr>
                        <td>{attack_name.capitalize()}</td>
                        <td>{auc:.3f}</td>
                        <td>{acc:.3f}</td>
                        <td class="{risk_class}">{risk_text}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </div>
        """
        
        # Add DP section if available
        if dp_report:
            html_content += f"""
        <div class="report-section">
            <h2>🛡️ Differential Privacy Analysis</h2>
            <img src="{dp_report.name}" alt="DP Report" class="report-image">
        </div>
            """
        
        # Add recommendations
        best_auc = max([results.get('auc', 0.5) for results in mia_results.values()])
        overfitting_gap = training_history.get('train_accuracies', [0])[-1] - training_history.get('test_accuracies', [0])[-1]
        
        html_content += f"""
        <div class="metadata">
            <h2>💡 Recommendations</h2>
            <div style="text-align: left;">
        """
        
        if best_auc > 0.8:
            html_content += """
                <h3 class="risk-high">🔴 CRITICAL PRIVACY RISK</h3>
                <ul>
                    <li><strong>Immediate Action Required:</strong> Apply differential privacy mechanisms</li>
                    <li><strong>Model Architecture:</strong> Reduce model complexity and capacity</li>
                    <li><strong>Data Access:</strong> Implement strict access controls and audit logs</li>
                    <li><strong>Deployment:</strong> Consider federated learning or secure aggregation</li>
                </ul>
            """
        elif best_auc > 0.7:
            html_content += """
                <h3 class="risk-high">🟠 HIGH PRIVACY RISK</h3>
                <ul>
                    <li><strong>Privacy Techniques:</strong> Consider differential privacy or other privacy-preserving methods</li>
                    <li><strong>Monitoring:</strong> Implement continuous privacy monitoring</li>
                    <li><strong>Access Control:</strong> Limit model and data access</li>
                    <li><strong>Regular Audits:</strong> Conduct periodic privacy assessments</li>
                </ul>
            """
        elif best_auc > 0.6:
            html_content += """
                <h3 class="risk-medium">🟡 MODERATE PRIVACY RISK</h3>
                <ul>
                    <li><strong>Basic Privacy Measures:</strong> Implement standard privacy practices</li>
                    <li><strong>Access Controls:</strong> Monitor and log model access</li>
                    <li><strong>Regular Monitoring:</strong> Periodic privacy assessments recommended</li>
                </ul>
            """
        else:
            html_content += """
                <h3 class="risk-low">🟢 LOW PRIVACY RISK</h3>
                <ul>
                    <li><strong>Standard Practices:</strong> Current security measures appear adequate</li>
                    <li><strong>Monitoring:</strong> Continue regular privacy monitoring</li>
                    <li><strong>Best Practices:</strong> Maintain current privacy-conscious approach</li>
                </ul>
            """
        
        if overfitting_gap > 30:
            html_content += """
                <h3>📉 Overfitting Mitigation</h3>
                <ul>
                    <li><strong>Training:</strong> Reduce number of training epochs</li>
                    <li><strong>Regularization:</strong> Add L1/L2 regularization, dropout, or batch normalization</li>
                    <li><strong>Data:</strong> Increase training dataset size if possible</li>
                    <li><strong>Architecture:</strong> Consider simpler model architectures</li>
                </ul>
            """
        
        html_content += """
            </div>
        </div>
        
        <div class="footer">
            <p>This report was automatically generated by the ML Privacy Analysis System.</p>
            <p>For questions or concerns about privacy risks, consult with your security team.</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Save HTML report
        html_path = self.output_dir / f'{save_name}.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path 