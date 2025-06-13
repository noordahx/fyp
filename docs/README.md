# Membership Inference Attacks and Differential Privacy Protection - Documentation

## Overview

This documentation covers a comprehensive framework for evaluating membership inference attacks (MIA) against machine learning models and implementing differential privacy (DP) protection mechanisms. The project provides end-to-end tools for training target models, conducting various types of membership inference attacks, and evaluating privacy-preserving defenses.

## 🏗️ Project Architecture

```
fyp/
├── scripts/                           # Main execution scripts
│   ├── train_target_model.py         # Target model training
│   ├── train_shadow_models.py        # Shadow models for attacks
│   ├── membership_inference_attacks.py # Comprehensive MIA suite
│   └── train_with_differential_privacy.py # DP protection methods
├── mia_lib/                           # Core library modules
│   ├── attack/                        # Attack implementations
│   ├── models.py                      # Model architectures
│   ├── data.py                        # Data handling
│   └── utils.py                       # Utility functions
├── differential_privacy/              # DP mechanisms and algorithms
│   └── dp_mechanisms.py              # DP implementation
├── evaluation/                        # Evaluation and metrics
│   └── comprehensive_metrics.py      # Advanced evaluation tools
├── configs/                           # Configuration files
│   └── mia_config.yaml               # Main configuration
└── docs/                             # Documentation (this directory)
    ├── target_model_training.md      # Target model documentation
    ├── shadow_models_training.md     # Shadow models documentation
    ├── membership_inference_attacks.md # MIA attacks documentation
    ├── differential_privacy_implementation.md # DP documentation
    └── README.md                     # This file
```

## 📊 Experimental Pipeline

The framework implements a complete experimental pipeline:

```
1. Target Model Training → 2. Shadow Models Training → 3. Attack Execution → 4. Privacy Protection → 5. Evaluation
         ↓                        ↓                      ↓                   ↓                    ↓
   Ground Truth Data        Attack Training Data    Attack Results    Protected Models      Comprehensive Metrics
```

### Phase 1: Target Model Training
- Trains the victim model that attackers will target
- Creates member/non-member ground truth labels
- Establishes baseline performance metrics
- **Documentation**: [target_model_training.md](target_model_training.md)

### Phase 2: Shadow Models Training  
- Trains multiple shadow models to simulate target model behavior
- Implements anti-overfitting techniques for realistic scenarios
- Generates attack training datasets
- **Documentation**: [shadow_models_training.md](shadow_models_training.md)

### Phase 3: Attack Execution
- Implements 5 different membership inference attack methods
- Provides comprehensive evaluation metrics
- Supports comparative analysis across attack types
- **Documentation**: [membership_inference_attacks.md](membership_inference_attacks.md)

### Phase 4: Privacy Protection
- Implements differential privacy mechanisms (DP-SGD, PATE, Output Perturbation)
- Provides privacy-utility tradeoff analysis
- Includes mathematical foundations and practical implementations
- **Documentation**: [differential_privacy_implementation.md](differential_privacy_implementation.md)

### Phase 5: Evaluation
- Comprehensive metrics for both model performance and privacy protection
- Statistical analysis and visualization tools
- Privacy risk assessment and recommendations
- **Documentation**: [comprehensive_metrics.py](../evaluation/comprehensive_metrics.py)

## 🚀 Quick Start Guide

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
pip install catboost tqdm pyyaml
```

### Basic Usage

1. **Train Target Model**:
```bash
cd fyp
python scripts/train_target_model.py
```

2. **Train Shadow Models**:
```bash
python scripts/train_shadow_models.py
```

3. **Execute Membership Inference Attacks**:
```bash
python scripts/membership_inference_attacks.py
```

4. **Apply Differential Privacy Protection**:
```bash
python scripts/train_with_differential_privacy.py
```

### Complete Pipeline
```bash
# Run all components sequentially
python scripts/train_target_model.py && \
python scripts/train_shadow_models.py && \
python scripts/membership_inference_attacks.py && \
python scripts/train_with_differential_privacy.py
```

## 🎯 Attack Methods Implemented

### 1. Shadow Model Attack (Attack S)
- **Method**: Train multiple shadow models to mimic target behavior
- **Strength**: High accuracy when shadow models are representative
- **Use Case**: Attacker has access to similar training data
- **Implementation**: CatBoost classifier on confidence features

### 2. Reference Model Attack (Attack R)  
- **Method**: Compare target outputs with reference models
- **Strength**: No need for shadow training data with membership labels
- **Use Case**: Attacker can train models on similar data distribution
- **Implementation**: Statistical distance metrics

### 3. Distillation Attack (Attack D)
- **Method**: Use knowledge distillation to create surrogate model
- **Strength**: Leverages model-specific patterns
- **Use Case**: Attacker has access to unlabeled data from similar distribution
- **Implementation**: Teacher-student distillation with distance metrics

### 4. Leave-One-Out Attack (Attack L)
- **Method**: Retrain model without each test sample
- **Strength**: Highest theoretical accuracy
- **Use Case**: Research scenarios with computational resources
- **Implementation**: Influence function approximation

### 5. Population Attack (Attack P)
- **Method**: Use population statistics without auxiliary models
- **Strength**: Requires minimal attacker resources
- **Use Case**: Simple baseline attack
- **Implementation**: Confidence thresholding

## 🛡️ Defense Mechanisms

### 1. DP-SGD (Differentially Private Stochastic Gradient Descent)
- **Mathematical Foundation**: Gradient clipping + Gaussian noise
- **Privacy Guarantee**: (ε, δ)-differential privacy
- **Best For**: Training phase protection
- **Trade-off**: Training complexity vs privacy strength

### 2. PATE (Private Aggregation of Teacher Ensembles)
- **Mathematical Foundation**: Private voting among teacher models  
- **Privacy Guarantee**: Pure differential privacy per query
- **Best For**: Scenarios with disjoint training data
- **Trade-off**: Model complexity vs privacy efficiency

### 3. Output Perturbation
- **Mathematical Foundation**: Laplace/Gaussian noise on outputs
- **Privacy Guarantee**: Configurable (ε, δ)-DP
- **Best For**: Inference phase protection
- **Trade-off**: Prediction accuracy vs privacy level

## 📈 Evaluation Metrics

### Model Performance Metrics
- **Accuracy, Precision, Recall, F1-Score**: Standard classification metrics
- **ROC AUC**: Area under receiver operating characteristic curve
- **Confidence Statistics**: Distribution analysis of prediction confidence
- **Per-Class Performance**: Class-specific evaluation metrics

### Attack Success Metrics
- **Attack Accuracy**: Overall attack success rate
- **Attack Advantage**: Improvement over random guessing
- **ROC AUC**: Attack discriminative power
- **TPR@FPR**: True positive rate at specific false positive rates
- **Balanced Accuracy**: Performance accounting for class imbalance

### Privacy Protection Metrics
- **Privacy Gain**: Reduction in attack success rate
- **Utility Loss**: Reduction in model performance
- **Privacy-Utility Ratio**: Efficiency of privacy protection
- **Privacy Budget Consumption**: DP parameter tracking

## 📋 Configuration

The system uses YAML configuration files for easy customization:

```yaml
# Example configuration structure
CFG:
  seed: 42
  target_model:
    architecture: resnet18
    epochs: 100
    learning_rate: 0.0005
    target_train_size: 7500
    
CFG_ATTACK:
  shadow:
    num_shadow_models: 32
    shadow_train_size: 3000
  attack:
    output_dim: 10
    catboost:
      iterations: 200
      depth: 2
```

**Configuration Documentation**: See [mia_config.yaml](../configs/mia_config.yaml) for full options.

## 🔬 Research Applications

### Academic Research
- **Privacy Risk Assessment**: Quantify membership inference vulnerabilities
- **Defense Evaluation**: Benchmark privacy-preserving techniques  
- **Algorithm Development**: Test new attack methods or defenses
- **Comparative Studies**: Systematic comparison across methods

### Industry Applications
- **Privacy Auditing**: Assess production model privacy risks
- **Compliance**: Meet privacy regulations (GDPR, CCPA)
- **Risk Management**: Quantify and mitigate privacy exposure
- **Defense Deployment**: Implement production privacy protection

## 📊 Interpreting Results

### Attack Success Interpretation
- **Accuracy > 0.8**: High privacy risk, strong defense needed
- **Accuracy 0.6-0.8**: Medium risk, moderate defenses recommended  
- **Accuracy 0.5-0.6**: Low risk, basic protections sufficient
- **Accuracy ≈ 0.5**: Minimal risk, random guessing performance

### Privacy-Utility Tradeoff
- **High Privacy Gain, Low Utility Loss**: Effective defense
- **Low Privacy Gain, High Utility Loss**: Inefficient defense
- **Balanced Trade-off**: Practical deployment consideration
- **Privacy Budget**: Monitor consumption for sustainable protection

### Differential Privacy Parameters
- **ε < 1.0**: Strong privacy (recommended for sensitive data)
- **ε = 1.0-5.0**: Moderate privacy (common in practice)
- **ε > 5.0**: Weak privacy (minimal protection)
- **δ ≤ 1/n²**: Standard delta selection (n = dataset size)

## 🛠️ Advanced Usage

### Custom Attack Implementation
```python
from scripts.membership_inference_attacks import MembershipInferenceAttackSuite

class CustomAttack:
    def attack_samples(self, model, data_loader):
        # Implement custom attack logic
        pass

# Integrate with existing framework
attack_suite = MembershipInferenceAttackSuite()
attack_suite.custom_attacks['my_attack'] = CustomAttack()
```

### Custom Defense Implementation  
```python
from differential_privacy.dp_mechanisms import DPMechanism

class CustomDPMechanism(DPMechanism):
    def add_noise(self, value, sensitivity):
        # Implement custom DP mechanism
        pass

# Use in training pipeline
mechanism = CustomDPMechanism(epsilon=1.0, delta=1e-5)
```

### Automated Hyperparameter Tuning
```python
from itertools import product

# Define parameter grids
privacy_levels = ['strict', 'moderate', 'relaxed']
noise_multipliers = [1.0, 1.5, 2.0]

# Grid search
results = {}
for privacy, noise in product(privacy_levels, noise_multipliers):
    result = train_dp_model(privacy_level=privacy, noise_multiplier=noise)
    results[(privacy, noise)] = result
```

## 🚨 Common Issues and Solutions

### Memory Issues
```bash
# Reduce batch sizes in config
train_batch_size: 64  # Instead of 128
eval_batch_size: 128  # Instead of 256

# Use gradient accumulation
accumulation_steps: 4
```

### Poor Attack Performance
```python
# Check target model overfitting
member_acc = evaluate_on_members(target_model)
non_member_acc = evaluate_on_non_members(target_model)
if member_acc - non_member_acc < 0.05:
    print("Target model needs more overfitting for realistic attacks")
```

### DP Training Convergence
```yaml
# Adjust DP-SGD parameters
dp_sgd:
  learning_rate: 0.15      # Often higher than non-private
  clip_norm: 1.0           # May need adjustment
  noise_multiplier: 1.1    # Start conservative
  epochs: 60              # Often need more epochs
```

## 📚 Extended Reading

### Theoretical Background
- **Differential Privacy**: Dwork & Roth (2014) - Algorithmic Foundations
- **Membership Inference**: Shokri et al. (2017) - Original attack paper
- **Privacy-Utility Tradeoffs**: Yeom et al. (2018) - Connection to overfitting

### Implementation References  
- **DP-SGD**: Abadi et al. (2016) - Deep learning with differential privacy
- **PATE**: Papernot et al. (2017) - Scalable private learning
- **Advanced Composition**: Dwork et al. (2010) - Boosting and differential privacy

### Recent Advances
- **Concentrated DP**: Bun & Steinke (2016) - Simplified analysis
- **RDP Accounting**: Mironov (2017) - Rényi differential privacy
- **Privacy Amplification**: Balle et al. (2018) - Sampling and shuffling

## 🤝 Contributing

### Code Contributions
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-attack`
3. Implement changes with tests
4. Update documentation
5. Submit pull request

### Documentation Contributions
1. Identify documentation gaps
2. Follow existing documentation style
3. Include mathematical foundations where applicable
4. Provide practical examples
5. Update this README if needed

### Research Contributions
1. Implement new attack methods
2. Add novel defense mechanisms  
3. Improve evaluation metrics
4. Enhance privacy accounting
5. Contribute benchmark datasets

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@software{mia_dp_framework,
  title={Comprehensive Membership Inference Attack and Differential Privacy Framework},
  author={[Author Name]},
  year={2024},
  url={[Repository URL]}
}
```

## 📞 Support

### Getting Help
- **Documentation**: Start with component-specific docs in this directory
- **Issues**: Check common issues section above
- **Examples**: See usage examples in each documentation file
- **Configuration**: Review [mia_config.yaml](../configs/mia_config.yaml)

### Reporting Bugs
1. Check existing issues first
2. Provide minimal reproduction example
3. Include configuration details
4. Specify environment information
5. Describe expected vs actual behavior

## 🔮 Future Enhancements

### Planned Features
- **Federated Learning Support**: MIA in federated settings
- **Graph Neural Networks**: Privacy attacks on graph models
- **Continual Learning**: Privacy in lifelong learning
- **Multi-Modal Models**: Attacks on vision-language models
- **Hardware Acceleration**: GPU-optimized DP training

### Research Directions
- **Adaptive Attacks**: Attacks robust to defenses
- **Privacy Amplification**: Better composition bounds
- **Utility Optimization**: Improved privacy-utility tradeoffs
- **Personalized Privacy**: Individual privacy preferences
- **Privacy-Preserving Federated Learning**: Cross-device privacy

---

## 📖 Documentation Index

| Component | Documentation | Purpose |
|-----------|--------------|---------|
| **Target Model** | [target_model_training.md](target_model_training.md) | Victim model training and evaluation |
| **Shadow Models** | [shadow_models_training.md](shadow_models_training.md) | Attack data generation and anti-overfitting |
| **MIA Attacks** | [membership_inference_attacks.md](membership_inference_attacks.md) | Comprehensive attack suite |
| **Differential Privacy** | [differential_privacy_implementation.md](differential_privacy_implementation.md) | Privacy protection mechanisms |
| **Evaluation** | [../evaluation/comprehensive_metrics.py](../evaluation/comprehensive_metrics.py) | Advanced metrics and analysis |

**Last Updated**: December 2024  
**Framework Version**: 1.0  
**Python Compatibility**: 3.8+  
**PyTorch Compatibility**: 1.12+