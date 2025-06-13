#!/usr/bin/env python3
"""
Demo script for the ML Privacy Analysis Visualization System.

This script demonstrates the visualization capabilities with different
scenarios including high-risk and low-risk models.

Usage:
    python scripts/demo_visualization.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mia_lib.visualization import MIAVisualizer

def create_demo_data():
    """Create demo training and MIA results for visualization."""
    
    # Scenario 1: High-risk overfitted model
    high_risk_training = {
        'train_accuracies': [25, 45, 65, 80, 90, 95, 98, 99, 99.5, 100],
        'test_accuracies': [30, 50, 60, 65, 68, 70, 71, 71.5, 71.8, 72],
        'train_losses': [2.3, 1.8, 1.2, 0.8, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01],
        'test_losses': [2.2, 1.6, 1.1, 0.9, 0.8, 0.75, 0.73, 0.72, 0.71, 0.70]
    }
    
    # High-risk MIA results (successful attacks)
    np.random.seed(42)
    n_samples = 1000
    
    # Threshold attack - clear separation
    member_scores = np.random.normal(0.95, 0.05, n_samples//2)
    non_member_scores = np.random.normal(0.75, 0.1, n_samples//2)
    threshold_probs = np.concatenate([member_scores, non_member_scores])
    threshold_labels = np.concatenate([np.ones(n_samples//2), np.zeros(n_samples//2)])
    threshold_preds = (threshold_probs > 0.85).astype(int)
    
    high_risk_mia = {
        'threshold': {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1': 0.85,
            'auc': 0.89,
            'probabilities': threshold_probs,
            'true_labels': threshold_labels,
            'predictions': threshold_preds
        },
        'loss': {
            'accuracy': 0.82,
            'precision': 0.80,
            'recall': 0.85,
            'f1': 0.82,
            'auc': 0.87,
            'probabilities': np.random.normal(0.8, 0.15, n_samples),
            'true_labels': np.concatenate([np.ones(n_samples//2), np.zeros(n_samples//2)]),
            'predictions': np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
        }
    }
    
    high_risk_model = {
        'model_name': 'High_Risk_Model',
        'dataset': 'CIFAR-10',
        'model_size': 'Large',
        'total_params': 5000000,
        'train_size': 10000,
        'test_size': 2000,
        'epochs': 100,
        'batch_size': 256,
        'lr': 0.001,
        'weight_decay': 0.0,
        'device': 'CUDA',
        'training_time': '45.2s'
    }
    
    # Scenario 2: Low-risk well-regularized model
    low_risk_training = {
        'train_accuracies': [25, 45, 65, 75, 82, 86, 88, 89, 89.5, 90],
        'test_accuracies': [30, 50, 60, 70, 78, 82, 84, 85, 85.5, 86],
        'train_losses': [2.3, 1.8, 1.2, 0.9, 0.7, 0.6, 0.55, 0.52, 0.50, 0.48],
        'test_losses': [2.2, 1.6, 1.1, 0.9, 0.75, 0.65, 0.60, 0.58, 0.56, 0.54]
    }
    
    # Low-risk MIA results (failed attacks)
    member_scores_low = np.random.normal(0.65, 0.1, n_samples//2)
    non_member_scores_low = np.random.normal(0.62, 0.1, n_samples//2)
    threshold_probs_low = np.concatenate([member_scores_low, non_member_scores_low])
    threshold_preds_low = (threshold_probs_low > 0.63).astype(int)
    
    low_risk_mia = {
        'threshold': {
            'accuracy': 0.52,
            'precision': 0.51,
            'recall': 0.53,
            'f1': 0.52,
            'auc': 0.54,
            'probabilities': threshold_probs_low,
            'true_labels': threshold_labels,
            'predictions': threshold_preds_low
        },
        'loss': {
            'accuracy': 0.49,
            'precision': 0.48,
            'recall': 0.51,
            'f1': 0.49,
            'auc': 0.51,
            'probabilities': np.random.normal(0.5, 0.05, n_samples),
            'true_labels': np.concatenate([np.ones(n_samples//2), np.zeros(n_samples//2)]),
            'predictions': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        }
    }
    
    low_risk_model = {
        'model_name': 'Low_Risk_Model',
        'dataset': 'CIFAR-10',
        'model_size': 'Medium',
        'total_params': 1500000,
        'train_size': 40000,
        'test_size': 10000,
        'epochs': 50,
        'batch_size': 128,
        'lr': 0.001,
        'weight_decay': 0.01,
        'device': 'CUDA',
        'training_time': '120.5s'
    }
    
    return {
        'high_risk': {
            'training_history': high_risk_training,
            'mia_results': high_risk_mia,
            'model_info': high_risk_model
        },
        'low_risk': {
            'training_history': low_risk_training,
            'mia_results': low_risk_mia,
            'model_info': low_risk_model
        }
    }

def create_dp_demo_data():
    """Create demo differential privacy results."""
    
    # Simulate DP results for different epsilon values
    epsilons = ['inf', '10.0', '5.0', '1.0', '0.5']
    model_accs = [0.92, 0.90, 0.87, 0.82, 0.75]
    attack_aucs = [0.85, 0.78, 0.70, 0.62, 0.55]
    attack_accs = [0.82, 0.75, 0.68, 0.60, 0.54]
    
    dp_results = {}
    for eps, model_acc, attack_auc, attack_acc in zip(epsilons, model_accs, attack_aucs, attack_accs):
        dp_results[eps] = {
            'model_accuracy': model_acc,
            'attack_auc': attack_auc,
            'attack_accuracy': attack_acc
        }
    
    return dp_results

def main():
    """Run the visualization demo."""
    
    print("🎨 ML Privacy Analysis Visualization Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("demo_reports")
    visualizer = MIAVisualizer(output_dir)
    
    # Generate demo data
    print("📊 Generating demo data...")
    demo_data = create_demo_data()
    dp_data = create_dp_demo_data()
    
    # Generate reports for high-risk scenario
    print("\n🔴 Generating HIGH RISK scenario reports...")
    high_risk = demo_data['high_risk']
    
    training_report = visualizer.create_training_report(
        high_risk['training_history'], 
        high_risk['model_info'], 
        "high_risk_training"
    )
    print(f"✓ Training report: {training_report}")
    
    mia_report = visualizer.create_mia_report(
        high_risk['mia_results'], 
        high_risk['model_info'], 
        "high_risk_mia"
    )
    print(f"✓ MIA report: {mia_report}")
    
    dashboard = visualizer.create_summary_dashboard(
        high_risk['training_history'], 
        high_risk['mia_results'], 
        high_risk['model_info'], 
        "high_risk_dashboard"
    )
    print(f"✓ Dashboard: {dashboard}")
    
    html_report = visualizer.generate_html_report(
        high_risk['training_history'], 
        high_risk['mia_results'], 
        high_risk['model_info'], 
        save_name="high_risk_complete"
    )
    print(f"✓ HTML report: {html_report}")
    
    # Generate reports for low-risk scenario
    print("\n🟢 Generating LOW RISK scenario reports...")
    low_risk = demo_data['low_risk']
    
    training_report = visualizer.create_training_report(
        low_risk['training_history'], 
        low_risk['model_info'], 
        "low_risk_training"
    )
    print(f"✓ Training report: {training_report}")
    
    mia_report = visualizer.create_mia_report(
        low_risk['mia_results'], 
        low_risk['model_info'], 
        "low_risk_mia"
    )
    print(f"✓ MIA report: {mia_report}")
    
    dashboard = visualizer.create_summary_dashboard(
        low_risk['training_history'], 
        low_risk['mia_results'], 
        low_risk['model_info'], 
        "low_risk_dashboard"
    )
    print(f"✓ Dashboard: {dashboard}")
    
    html_report = visualizer.generate_html_report(
        low_risk['training_history'], 
        low_risk['mia_results'], 
        low_risk['model_info'], 
        save_name="low_risk_complete"
    )
    print(f"✓ HTML report: {html_report}")
    
    # Generate differential privacy report
    print("\n🛡️ Generating DIFFERENTIAL PRIVACY report...")
    dp_report = visualizer.create_dp_report(dp_data, "dp_analysis")
    print(f"✓ DP report: {dp_report}")
    
    # Generate comprehensive comparison report
    print("\n📋 Generating COMPARISON report...")
    comparison_html = visualizer.generate_html_report(
        high_risk['training_history'], 
        high_risk['mia_results'], 
        high_risk['model_info'], 
        dp_results=dp_data,
        save_name="comprehensive_analysis"
    )
    print(f"✓ Comprehensive report: {comparison_html}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 All reports saved to: {output_dir}")
    print(f"\n📖 Report Types Generated:")
    print(f"   • Training Analysis: Shows overfitting progression and risk assessment")
    print(f"   • MIA Attack Analysis: Detailed attack performance and vulnerabilities")
    print(f"   • Executive Dashboard: High-level summary with recommendations")
    print(f"   • HTML Reports: Complete interactive analysis")
    print(f"   • Differential Privacy: Privacy-utility tradeoff analysis")
    
    print(f"\n🌐 Open these HTML reports in your browser:")
    print(f"   • High Risk Model: {output_dir}/high_risk_complete.html")
    print(f"   • Low Risk Model: {output_dir}/low_risk_complete.html")
    print(f"   • Comprehensive Analysis: {output_dir}/comprehensive_analysis.html")
    
    print(f"\n💡 Key Features Demonstrated:")
    print(f"   ✓ Risk color coding (Red=Critical, Orange=High, Yellow=Moderate, Green=Low)")
    print(f"   ✓ ROC curves with proper data handling")
    print(f"   ✓ Score distribution analysis")
    print(f"   ✓ Overfitting progression tracking")
    print(f"   ✓ Actionable privacy recommendations")
    print(f"   ✓ Professional HTML reports suitable for presentations")

if __name__ == '__main__':
    main() 