# Complete DP Training & MIA Evaluation Pipeline

## 🎯 Overview

This pipeline will automatically:

1. **Train 6 target models** with different privacy methods:
   - Standard (no privacy)
   - DP-SGD with ε = 3.0, 1.0, 0.5
   - Output Perturbation with ε = 3.0, 1.0

2. **Run 4 MIA attacks** on each model:
   - Threshold Attack (confidence-based)
   - Loss Attack (cross-entropy loss-based)  
   - Population Attack (statistical distance-based)
   - Shadow Attack (surrogate model-based)

3. **Generate comprehensive results** saved to `colab_results/`

## 🚀 Quick Start

### Step 1: Run the Complete Pipeline
```bash
# Recommended: Fast evaluation (1.5-2 hours)
python run_complete_evaluation.py --config configs/evaluation_config.yaml

# Or comprehensive evaluation (2.5-3.5 hours)
python run_complete_evaluation.py --config configs/full_config.yaml
```

### Step 2: Check Results
```bash
# View summary table
cat colab_results/attack_results_summary.csv

# Read full report  
cat colab_results/final_report.txt

# Check training logs
cat colab_results/logs/evaluation_*.log
```

## 📁 Output Structure

```
colab_results/
├── models/                           # Trained models
│   ├── standard/
│   ├── dp_sgd_eps3/
│   ├── dp_sgd_eps1/
│   ├── dp_sgd_eps0_5/
│   ├── output_pert_eps3/
│   └── output_pert_eps1/
├── attacks/                          # Attack results
│   ├── standard/
│   │   ├── threshold_results.json
│   │   ├── loss_results.json
│   │   ├── population_results.json
│   │   └── shadow_results.json
│   └── ... (for each model)
├── attack_results_summary.csv        # 📊 MAIN RESULTS TABLE
├── attack_results_summary.json       # Raw results data
├── training_summary.json             # Training status
├── final_report.txt                  # 📋 COMPREHENSIVE REPORT
└── logs/                            # Execution logs
    └── evaluation_*.log
```

## 📊 Key Results Files

### `attack_results_summary.csv`
Main results table with columns:
- `model_name`: Privacy method used
- `model_epsilon`: Privacy budget
- `attack_type`: Attack method
- `auc`: Area under ROC curve (higher = more vulnerable)
- `accuracy`: Attack accuracy (higher = more vulnerable) 
- `attack_advantage`: Privacy leakage metric

### `final_report.txt`
Human-readable summary with:
- Training success/failure status
- Attack results organized by model
- Privacy analysis and interpretation

## 🎛️ Advanced Usage

### Run Only Specific Phases
```bash
# Only train models (skip attacks)
python run_complete_evaluation.py --skip-attacks

# Only run attacks (requires existing models)
python run_complete_evaluation.py --skip-training

# Custom output directory
python run_complete_evaluation.py --output-dir ./my_results
```

### Run Individual Components
```bash
# Train single model
python scripts/train.py --config configs/evaluation_config.yaml --method dp_sgd_custom --epsilon 1.0 --output_dir ./test_model

# Run single attack
python scripts/attack.py --config configs/evaluation_config.yaml --model ./test_model/cifar10_dp_sgd_custom_eps1.0.pt --attack shadow --output-dir ./test_attack
```

## 📈 Expected Results

### Privacy Ranking (Most to Least Vulnerable)
1. **Standard**: AUC ~0.8-0.9, Acc ~80-90% (No privacy protection)
2. **DP-SGD ε=3.0**: AUC ~0.6-0.7, Acc ~60-70% (Low privacy)
3. **Output Pert ε=3.0**: AUC ~0.6-0.7, Acc ~60-70% (Variable)
4. **DP-SGD ε=1.0**: AUC ~0.55-0.65, Acc ~55-65% (Medium privacy)
5. **Output Pert ε=1.0**: AUC ~0.55-0.65, Acc ~55-65% (Variable)
6. **DP-SGD ε=0.5**: AUC ~0.5-0.6, Acc ~50-60% (High privacy)

### Attack Effectiveness Ranking
1. **Shadow Attack**: Most sophisticated, highest AUC
2. **Loss Attack**: Very effective, second highest AUC
3. **Population Attack**: Moderate effectiveness
4. **Threshold Attack**: Baseline, lowest AUC

## 🛠️ Configuration Options

### `evaluation_config.yaml` (Recommended)
- **Training**: 20 epochs, smaller batch sizes
- **Shadow Attack**: 4 models, 8 epochs each
- **Runtime**: ~1.5-2 hours
- **Quality**: Good for research and testing

### `full_config.yaml` (Comprehensive)
- **Training**: 30 epochs, larger configurations
- **Shadow Attack**: 8 models, 10 epochs each  
- **Runtime**: ~2.5-3.5 hours
- **Quality**: Highest attack performance

## 🔍 Interpreting Results

### AUC (Area Under Curve)
- **0.5**: Random guessing (perfect privacy)
- **0.6-0.7**: Moderate privacy leakage
- **0.7-0.8**: Significant privacy leakage
- **0.8+**: High privacy leakage (poor privacy)

### Attack Advantage
- **0.0**: No attack capability
- **0.2-0.4**: Moderate privacy risk
- **0.4+**: High privacy risk

### Privacy-Utility Trade-off
- Lower ε (epsilon) = Better privacy but potentially lower model accuracy
- Higher ε = Worse privacy but better model utility
- Goal: Find sweet spot balancing privacy and utility

## ⚠️ Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch sizes in config
2. **Training fails**: Check model paths and permissions
3. **Attacks fail**: Verify model files exist and are readable
4. **Long runtime**: Use `evaluation_config.yaml` instead

### Debug Mode
```bash
# Check pipeline without running
python test_evaluation_pipeline.py

# Monitor progress
tail -f colab_results/logs/evaluation_*.log
```

## 🎉 You're Ready!

Run the complete evaluation with:
```bash
python run_complete_evaluation.py --config configs/evaluation_config.yaml
```

Results will be saved to `colab_results/` folder. Check `attack_results_summary.csv` for the main findings!