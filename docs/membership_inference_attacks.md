# Membership Inference Attacks Documentation

## Overview

The membership inference attacks script (`scripts/membership_inference_attacks.py`) implements a comprehensive suite of membership inference attack (MIA) methods against machine learning models. This script serves as the core evaluation component for assessing privacy vulnerabilities of trained models by attempting to determine whether specific data points were used during training.

## Purpose

Membership inference attacks aim to answer the fundamental question: **"Was this data point used to train the model?"** This script implements multiple attack strategies to:

1. **Evaluate Privacy Vulnerabilities**: Assess how much information models leak about their training data
2. **Compare Attack Effectiveness**: Benchmark different attack methods against the same target
3. **Support Defense Development**: Provide baselines for evaluating privacy-preserving techniques
4. **Research Privacy Risks**: Systematic analysis of membership inference vulnerabilities

## Mathematical Foundation

### Membership Inference Problem

Given:
- A trained model `M: X → Y`
- A data point `(x, y)`
- Access to model outputs `M(x)`

**Goal**: Determine if `(x, y) ∈ D_train` where `D_train` is the training dataset.

### Attack Classification

Attacks can be categorized by:

1. **Knowledge Level**:
   - **Black-box**: Only access to model outputs
   - **White-box**: Access to model parameters/gradients

2. **Training Data**:
   - **Shadow-based**: Train auxiliary models on similar data
   - **Shadow-free**: Use only the target model

3. **Query Complexity**:
   - **Single-query**: One query per data point
   - **Multi-query**: Multiple queries for better accuracy

## Implemented Attack Methods

### 1. Shadow Model Attack (Attack S)

**Mathematical Foundation:**
```
Given shadow models {M₁, M₂, ..., Mₖ} trained on datasets {D₁, D₂, ..., Dₖ}
For each shadow model Mᵢ:
  - Member examples: {(f(Mᵢ(x)), 1) | x ∈ Dᵢ}
  - Non-member examples: {(f(Mᵢ(x)), 0) | x ∉ Dᵢ}

Train attack classifier: A(f(M(x))) → {0, 1}
```

**Feature Extraction:**
```python
f(M(x)) = [p₁, p₂, ..., pₙ]  # Output probabilities
# or
f(M(x)) = [max(p), entropy(p), ...]  # Derived features
```

**Implementation Details:**
- Uses CatBoost classifier for final attack model
- Combines data from all shadow models
- Extracts confidence-based features from model outputs

### 2. Reference Model Attack (Attack R)

**Mathematical Foundation:**
```
Train reference models {R₁, R₂, ..., Rₘ} on similar distribution
For test point x:
  distance = metric(M(x), [R₁(x), R₂(x), ..., Rₘ(x)])
  prediction = 1 if distance > threshold else 0
```

**Distance Metrics:**
- **L1 Distance**: `∑ᵢ |M(x)ᵢ - R̄(x)ᵢ|`
- **L2 Distance**: `√∑ᵢ (M(x)ᵢ - R̄(x)ᵢ)²`
- **KL Divergence**: `∑ᵢ M(x)ᵢ log(M(x)ᵢ / R̄(x)ᵢ)`

Where `R̄(x) = (1/m) ∑ⱼ Rⱼ(x)` is the average reference prediction.

### 3. Distillation Attack (Attack D)

**Mathematical Foundation:**
```
Train distilled model D on unlabeled data using target model M as teacher:
  L_distill = KL(D(x), M(x)) for x ~ unlabeled distribution

Attack metric: distance(M(x), D(x))
Intuition: Member data will have smaller distillation error
```

**Distillation Process:**
```python
# Teacher-student training
for x in unlabeled_data:
    teacher_output = M(x)  # Target model
    student_output = D(x)  # Distilled model
    loss = KL_divergence(student_output, teacher_output)
    optimize(D, loss)
```

### 4. Leave-One-Out Attack (Attack L)

**Mathematical Foundation:**
```
For test point x:
  1. Retrain model M' on D_train \ {x}
  2. Compare: loss_diff = L(M'(x)) - L(M(x))
  3. Predict: member if loss_diff > threshold

Intuition: Removing member data increases loss more than non-member data
```

**Computational Complexity:**
- Requires retraining for each test sample
- Most accurate but computationally expensive
- Often approximated using influence functions

### 5. Population Attack (Attack P)

**Mathematical Foundation:**
```
Use population statistics without additional models:
  - Confidence threshold: member if max(M(x)) > θ
  - Entropy threshold: member if H(M(x)) < θ
  - Loss threshold: member if L(M(x), y) < θ

Where H(p) = -∑ᵢ pᵢ log(pᵢ) is entropy
```

**Population Metrics:**
- **Prediction Confidence**: `max(M(x))`
- **Prediction Entropy**: `-∑ᵢ M(x)ᵢ log(M(x)ᵢ)`
- **Prediction Loss**: Cross-entropy with true label

## Implementation Details

### Class Structure

```python
class MembershipInferenceAttackSuite:
    """
    Comprehensive MIA attack suite with multiple methods
    """
```

### Key Methods

#### 1. `load_data_and_target_model()`
Loads the target model and prepares test data for attacks.

#### 2. `prepare_attack_test_data(test_size=1000)`
Creates balanced member/non-member test sets for evaluation.

#### 3. Attack-Specific Methods:
- `run_shadow_attack()`: Executes shadow model attack
- `run_reference_attack()`: Executes reference model attack  
- `run_distillation_attack()`: Executes distillation attack
- `run_leave_one_out_attack()`: Executes leave-one-out attack
- `run_population_attack()`: Executes population-based attack

#### 4. `evaluate_attack_performance()`
Comprehensive evaluation with multiple metrics.

#### 5. `create_comparative_analysis()`
Generates comparison plots and analysis across all attacks.

## Configuration Parameters

### Attack Configuration

```yaml
attack:
  # Shadow Model Attack (Attack S)
  output_dim: 10
  catboost:
    iterations: 200
    depth: 2
    learning_rate: 0.25
    loss_function: "Logloss"
    
  # Reference Model Attack (Attack R)
  R:
    num_reference_models: 3
    test_samples: 500
    threshold: 0.5
    method: "mean_prob_distance"
    
  # Distillation Attack (Attack D)
  D:
    distillation_subset_size: 1000
    epochs: 5
    temperature: 3.0
    alpha: 1.0
    lr: 0.001
    threshold: 0.5
    distance_metric: "l1"
    
  # Leave-One-Out Attack (Attack L)
  L:
    sample: 5                    # Number of samples (computational limit)
    threshold: 0.05
    retrain_epochs: 1
    
  # Population Attack (Attack P) - No additional parameters needed
```

### Test Data Configuration

```python
# Attack test data preparation
test_size = 1000                 # Total samples for attack evaluation
member_ratio = 0.5              # Balance between members/non-members
```

## Usage Instructions

### Basic Usage

```bash
cd fyp
python scripts/membership_inference_attacks.py
```

### Programmatic Usage

```python
from scripts.membership_inference_attacks import MembershipInferenceAttackSuite

# Initialize attack suite
attack_suite = MembershipInferenceAttackSuite("configs/mia_config.yaml")

# Run complete pipeline
results = attack_suite.run_complete_pipeline()

# Access individual attack results
shadow_results = results['attack_results']['shadow']
reference_results = results['attack_results']['reference']

# Print summary
print(f"Shadow attack accuracy: {shadow_results['accuracy']:.4f}")
print(f"Reference attack accuracy: {reference_results['accuracy']:.4f}")
```

### Running Individual Attacks

```python
# Initialize and setup
attack_suite = MembershipInferenceAttackSuite()
attack_suite.load_data_and_target_model()
attack_suite.prepare_attack_test_data()

# Run specific attacks
shadow_result = attack_suite.run_shadow_attack()
reference_result = attack_suite.run_reference_attack()

# Evaluate
shadow_metrics = attack_suite.evaluate_attack_performance('shadow', shadow_result)
print(f"Shadow attack ROC AUC: {shadow_metrics['roc_auc']:.4f}")
```

## Evaluation Metrics

### Primary Attack Metrics

#### 1. Classification Metrics
- **Accuracy**: Overall attack success rate
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **F1-Score**: Harmonic mean of precision and recall

#### 2. ROC Analysis
- **ROC AUC**: Area under ROC curve
- **TPR at FPR**: True positive rate at specific false positive rates
  - TPR@FPR=1%: High precision operating point
  - TPR@FPR=10%: Balanced operating point

#### 3. Attack-Specific Metrics
- **Attack Advantage**: `max(0, accuracy - 0.5)` (advantage over random guessing)
- **Balanced Accuracy**: `(TPR + TNR) / 2`
- **Privacy Risk Score**: Normalized attack success rate

### Advanced Evaluation

#### 1. Confidence Analysis
```python
member_confidences = get_attack_confidences(members)
non_member_confidences = get_attack_confidences(non_members)

confidence_gap = np.mean(member_confidences) - np.mean(non_member_confidences)
separability = statistical_distance(member_confidences, non_member_confidences)
```

#### 2. Statistical Significance
- **Kolmogorov-Smirnov Test**: Distribution differences
- **Mann-Whitney U Test**: Median differences  
- **Chi-square Test**: Independence testing

#### 3. Cross-Attack Correlation
```python
correlation_matrix = np.corrcoef([
    attack1_scores, attack2_scores, attack3_scores
])
```

## Output Files

### Directory Structure

```
models/attack_model/
├── mia_attacks_report.txt         # Comprehensive text report
├── attack_evaluation_metrics.json # Detailed metrics in JSON
├── mia_comparative_analysis.png   # Comparison visualizations
└── attack_specific_results/
    ├── shadow_attack_roc.png
    ├── reference_attack_dist.png
    └── ...
```

### Metrics File Structure

```json
{
  "shadow": {
    "attack_name": "shadow",
    "method": "Shadow Model Attack",
    "accuracy": 0.7850,
    "precision": 0.7642,
    "recall": 0.8123,
    "f1_score": 0.7875,
    "roc_auc": 0.8634,
    "attack_advantage": 0.2850,
    "tpr_at_fpr_0.01": 0.1234,
    "tpr_at_fpr_0.1": 0.6789
  },
  "reference": { ... },
  "distillation": { ... }
}
```

### Report Structure

```text
MEMBERSHIP INFERENCE ATTACKS - COMPREHENSIVE REPORT
================================================================

ATTACK SUMMARY:
- Shadow Model Attack: Uses multiple shadow models for training
- Reference Model Attack: Compares with reference model outputs  
- Distillation Attack: Uses knowledge distillation comparison
- Leave-One-Out Attack: Retrains model without each sample
- Population Attack: Uses population statistics

EVALUATION RESULTS:
SHADOW ATTACK:
  Accuracy: 0.7850
  Precision: 0.7642
  ROC AUC: 0.8634
  Attack Advantage: 0.2850
```

## Visualization

The script generates comprehensive comparison plots:

### 1. ROC Curves Comparison
Multi-attack ROC curves on the same plot showing:
- Individual attack performance
- Random baseline (diagonal line)
- AUC scores for each attack

### 2. Precision-Recall Curves
PR curves for attacks with sufficient probability outputs.

### 3. Attack Performance Comparison
Bar charts comparing:
- Accuracy across attacks
- ROC AUC across attacks  
- Attack advantage scores
- Precision vs recall trade-offs

### 4. Attack Correlation Heatmap
Correlation matrix showing how different attacks relate to each other.

### 5. Confidence Distribution Analysis
Histograms and box plots showing:
- Member vs non-member confidence distributions
- Attack score distributions
- Statistical separation metrics

### 6. Summary Dashboard
Comprehensive dashboard with:
- Performance summary table
- Key metrics comparison
- Best/worst performing attacks
- Recommendations

## Best Practices

### 1. Attack Selection Strategy

**For comprehensive evaluation:**
```python
# Run all attacks for complete picture
attacks_to_run = ['shadow', 'reference', 'distillation', 'population']

# Skip computationally expensive attacks for quick evaluation
quick_attacks = ['shadow', 'reference', 'population']
```

**For specific scenarios:**
- **Limited computational resources**: Population + Reference attacks
- **No auxiliary data**: Population attack only
- **Maximum accuracy**: Shadow + Leave-one-out attacks

### 2. Test Data Preparation

**Balanced evaluation:**
```python
test_size = 1000
member_ratio = 0.5  # Equal members and non-members
```

**Stratified sampling:**
```python
# Ensure class balance in both members and non-members
stratify_by_class = True
min_samples_per_class = 50
```

### 3. Threshold Selection

**ROC-based thresholds:**
```python
# Optimal threshold from ROC curve
optimal_threshold = threshold_at_max_f1(fpr, tpr, thresholds)

# High precision threshold
high_precision_threshold = threshold_at_fpr(fpr, tpr, thresholds, target_fpr=0.01)
```

**Cross-validation thresholds:**
```python
# Use validation set for threshold selection
validation_thresholds = cross_validate_thresholds(attack_scores, true_labels)
```

### 4. Statistical Validation

**Significance testing:**
```python
# Compare attack performance
p_value = statistical_significance_test(attack1_results, attack2_results)
is_significant = p_value < 0.05
```

**Confidence intervals:**
```python
# Bootstrap confidence intervals for metrics
ci_lower, ci_upper = bootstrap_confidence_interval(attack_accuracies, alpha=0.05)
```

## Troubleshooting

### Common Issues

#### 1. Poor Attack Performance
**Problem:** All attacks achieve ~50% accuracy (random performance)

**Potential Causes:**
- Target model well-regularized
- Insufficient overfitting in target model
- Poorly configured attack parameters

**Solutions:**
```python
# Check target model overfitting
member_acc = evaluate_on_members(target_model)
non_member_acc = evaluate_on_non_members(target_model)
overfitting_gap = member_acc - non_member_acc

if overfitting_gap < 0.05:
    print("Target model shows little overfitting - attacks may be ineffective")
    
# Verify attack data quality
check_attack_dataset_balance()
verify_member_non_member_labels()
```

#### 2. Shadow Attack Failure
**Problem:** Shadow attack fails to load or perform poorly

**Solutions:**
```python
# Check shadow models existence
shadow_models_path = "models/shadow_model/"
if not os.path.exists(shadow_models_path):
    print("Shadow models not found. Run shadow model training first.")
    
# Verify attack dataset
attack_dataset = pd.read_csv("models/shadow_model/attack_dataset.csv")
print(f"Attack dataset size: {len(attack_dataset)}")
print(f"Member ratio: {attack_dataset['label'].mean():.3f}")
```

#### 3. Memory Issues
**Problem:** CUDA out of memory during attack execution

**Solutions:**
```python
# Reduce batch sizes
eval_batch_size = 128  # Reduce from 256
test_size = 500       # Reduce test samples

# Process attacks sequentially
torch.cuda.empty_cache()
del unused_models
```

#### 4. Inconsistent Results
**Problem:** Different results across runs

**Solutions:**
```python
# Ensure reproducibility
seed_everything(42)
torch.backends.cudnn.deterministic = True

# Use fixed test splits
save_test_indices = True
load_fixed_test_split = True
```

### Performance Optimization

#### 1. Computational Efficiency
```python
# Parallel processing for independent attacks
from multiprocessing import Pool
with Pool(processes=4) as pool:
    results = pool.map(run_single_attack, attack_configs)
    
# GPU memory management
torch.cuda.empty_cache()
model.cpu()  # Move to CPU when not needed
```

#### 2. Attack-Specific Optimizations

**Shadow Attack:**
```python
# Use pre-computed features
if os.path.exists("attack_features.npy"):
    features = np.load("attack_features.npy")
else:
    features = extract_features(target_model, test_data)
    np.save("attack_features.npy", features)
```

**Reference Attack:**
```python
# Cache reference model outputs
reference_outputs = {}
for x in test_data:
    if x not in reference_outputs:
        reference_outputs[x] = compute_reference_outputs(x)
```

## Advanced Features

### 1. Adaptive Thresholding

```python
def adaptive_threshold_selection(attack_scores, method='f1'):
    """Select optimal threshold based on validation performance"""
    if method == 'f1':
        return threshold_at_max_f1(attack_scores)
    elif method == 'balanced':
        return threshold_at_balanced_accuracy(attack_scores)
    elif method == 'precision':
        return threshold_at_target_precision(attack_scores, target=0.9)
```

### 2. Ensemble Attacks

```python
def ensemble_attack(attack_results, weights=None):
    """Combine multiple attacks for improved performance"""
    if weights is None:
        weights = [1.0] * len(attack_results)
        
    ensemble_scores = np.zeros(len(attack_results[0]['predictions_proba']))
    for i, result in enumerate(attack_results):
        ensemble_scores += weights[i] * result['predictions_proba']
        
    ensemble_scores /= sum(weights)
    return ensemble_scores
```

### 3. Meta-Learning Attacks

```python
def meta_learning_attack(attack_features, attack_labels):
    """Learn which attack method works best for which samples"""
    meta_features = extract_meta_features(attack_features)
    meta_model = train_meta_classifier(meta_features, attack_labels)
    return meta_model
```

### 4. Transferability Analysis

```python
def analyze_attack_transferability(source_model, target_models, attack_method):
    """Analyze how attacks transfer across different models"""
    transferability_scores = {}
    
    for target_name, target_model in target_models.items():
        # Train attack on source model
        attack_model = train_attack(source_model, attack_method)
        
        # Test on target model
        target_accuracy = evaluate_attack(attack_model, target_model)
        transferability_scores[target_name] = target_accuracy
        
    return transferability_scores
```

## Security Considerations

### 1. Evaluation Integrity

**Prevent data leakage:**
```python
# Ensure complete separation of training and test data
assert len(set(train_indices) & set(test_indices)) == 0

# Use different random seeds for different components
np.random.seed(42)  # For data splitting
torch.manual_seed(43)  # For model training
```

**Realistic attack assumptions:**
```python
# Limit attacker knowledge
attacker_knowledge = {
    'model_architecture': False,  # Unknown to attacker
    'training_data_distribution': True,  # Reasonable assumption
    'hyperparameters': False,  # Unknown to attacker
    'training_procedure': False  # Unknown to attacker
}
```

### 2. Privacy Risk Assessment

**Risk categorization:**
```python
def categorize_privacy_risk(attack_accuracy):
    """Categorize privacy risk based on attack success"""
    if attack_accuracy > 0.9:
        return "CRITICAL"
    elif attack_accuracy > 0.8:
        return "HIGH" 
    elif attack_accuracy > 0.7:
        return "MEDIUM"
    elif attack_accuracy > 0.6:
        return "LOW"
    else:
        return "MINIMAL"
```

**Mitigation recommendations:**
```python
def recommend_mitigations(privacy_risk, attack_results):
    """Recommend privacy protection methods based on attack results"""
    recommendations = []
    
    if privacy_risk in ["CRITICAL", "HIGH"]:
        recommendations.extend([
            "Implement differential privacy with small epsilon (< 1.0)",
            "Use strong regularization techniques",
            "Consider federated learning approaches"
        ])
    elif privacy_risk == "MEDIUM":
        recommendations.extend([
            "Apply moderate differential privacy (epsilon 1-5)",
            "Use output perturbation",
            "Implement model ensemble techniques"
        ])
        
    return recommendations
```

## Integration with Defense Evaluation

### 1. Before-After Comparison

```python
def evaluate_defense_effectiveness(baseline_results, protected_results):
    """Compare attack success before and after applying defenses"""
    effectiveness = {}
    
    for attack_name in baseline_results:
        baseline_acc = baseline_results[attack_name]['accuracy']
        protected_acc = protected_results[attack_name]['accuracy']
        
        effectiveness[attack_name] = {
            'privacy_gain': baseline_acc - protected_acc,
            'relative_improvement': (baseline_acc - protected_acc) / baseline_acc,
            'remaining_risk': protected_acc
        }
        
    return effectiveness
```

### 2. Defense Robustness Testing

```python
def test_defense_robustness(defense_method, attack_variants):
    """Test defense against multiple attack variants"""
    robustness_scores = {}
    
    for variant_name, attack_config in attack_variants.items():
        # Apply defense
        protected_model = defense_method.protect(target_model)
        
        # Run attack variant
        attack_result = run_attack(protected_model, attack_config)
        robustness_scores[variant_name] = attack_result['accuracy']
        
    return robustness_scores
```

## Future Extensions

### 1. Multi-Modal Attacks
Support for attacks on models handling multiple data types (text, images, audio).

### 2. Federated Learning Attacks
Adaptation for membership inference in federated learning settings.

### 3. Continual Learning Attacks
Attacks against models that learn continuously over time.

### 4. Graph Neural Network Attacks
Specialized attacks for graph-based models.

### 5. Adversarial MIA
Attacks that are robust against adaptive defenses.

## References

1. Shokri, R., et al. "Membership inference attacks against machine learning models." 2017 IEEE Symposium on Security and Privacy (SP).

2. Salem, A., et al. "ML-Leaks: Model and data independent membership inference attacks and defenses on machine learning models." NDSS 2019.

3. Song, L., & Mittal, P. "Systematic evaluation of privacy risks of machine learning models." arXiv preprint arXiv:2003.10595 (2020).

4. Yeom, S., et al. "Privacy risk in machine learning: Analyzing the connection to overfitting." 2018 IEEE 31st Computer Security Foundations Symposium (CSF).

5. Nasr, M., et al. "Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning." 2019 IEEE symposium on security and privacy (SP).

6. Carlini, N., et al. "The secret sharer: Evaluating and testing unintended memorization in neural networks." 28th USENIX Security Symposium, 2019.

7. Chen, D., et al. "GAN-Leaks: A taxonomy of membership inference attacks against generative models." Proceedings of the 2020 ACM SIGSAC conference on computer and communications security.

# Membership Inference Attacks: Mathematical Foundations

This document explains the mathematical foundations of the membership inference attacks (MIA) implemented in this project.

## Overview

Membership inference attacks aim to determine whether a given data point was used to train a machine learning model. These attacks exploit the fact that models often behave differently on their training data compared to unseen data.

## Attack Methods

### 1. Shadow Model Attack (Attack S)

The shadow model attack trains multiple "shadow" models to mimic the behavior of the target model, then uses these to train an attack classifier.

**Mathematical Formulation:**

1. Train K shadow models {M₁, M₂, ..., Mₖ} on different subsets of data
2. For each shadow model Mᵢ:
   - Generate training data: Dᵢ = {(x, y) | x ∈ X, y = 1 if x ∈ Dᵢᵗʳᵃⁱⁿ else 0}
   - Train attack classifier A on Dᵢ
3. Final attack: A(x) = 1 if x was in target model's training data, 0 otherwise

**Key Insight:**
The attack exploits the difference in model confidence between training and non-training samples:
P(M(x) = y | x ∈ Dᵗʳᵃⁱⁿ) > P(M(x) = y | x ∉ Dᵗʳᵃⁱⁿ)

### 2. Reference Model Attack (Attack R)

The reference attack compares the target model's outputs with those of reference models trained on different data.

**Mathematical Formulation:**

1. Train K reference models {R₁, R₂, ..., Rₖ} on different data subsets
2. For input x, compute:
   - Target model output: pₜ = M(x)
   - Reference model outputs: pᵣ = (R₁(x) + R₂(x) + ... + Rₖ(x)) / K
3. Membership score: d(pₜ, pᵣ) where d is a distance metric (e.g., L1, L2, KL)

**Distance Metrics:**
- L1: d(p, q) = ∑|pᵢ - qᵢ|
- L2: d(p, q) = √∑(pᵢ - qᵢ)²
- KL: d(p, q) = ∑pᵢ log(pᵢ/qᵢ)
- JS: d(p, q) = 0.5[KL(p||m) + KL(q||m)] where m = 0.5(p + q)

### 3. Distillation Attack (Attack D)

The distillation attack uses model distillation to extract membership information.

**Mathematical Formulation:**

1. Train a student model S to mimic target model M:
   L(S, M) = ∑ᵪ KL(M(x) || S(x))
2. Use the distillation gap as membership signal:
   d(x) = KL(M(x) || S(x))
3. Membership prediction: d(x) < threshold

**Key Insight:**
The student model learns the target model's behavior better on training data than on non-training data.

### 4. Leave-One-Out Attack (Attack L)

The leave-one-out attack trains multiple models, each leaving out one training sample.

**Mathematical Formulation:**

1. For each training sample xᵢ:
   - Train model M₋ᵢ on D \ {xᵢ}
   - Compute prediction difference: Δᵢ = |M(xᵢ) - M₋ᵢ(xᵢ)|
2. Membership score: Δ(x) = minᵢ Δᵢ

**Key Insight:**
Training samples show larger prediction differences when left out compared to non-training samples.

### 5. Population Attack (Attack P)

The population attack uses population-level statistics to infer membership.

**Mathematical Formulation:**

1. Compute population statistics:
   - Mean: μ = E[M(x) | x ~ P]
   - Variance: σ² = Var[M(x) | x ~ P]
2. For input x, compute z-score:
   z(x) = (M(x) - μ) / σ
3. Membership prediction: |z(x)| > threshold

**Key Insight:**
Training samples often deviate more from population statistics than non-training samples.

## Evaluation Metrics

### 1. Attack Performance

- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)

### 2. ROC and AUC

- ROC curve plots True Positive Rate vs False Positive Rate
- AUC (Area Under Curve) measures attack's discriminative ability
- Higher AUC indicates better attack performance

### 3. Privacy Risk

- Membership advantage: |TPR - FPR|
- Higher advantage indicates greater privacy risk
- Perfect privacy: advantage = 0 (attack performs no better than random)

## Defenses

### 1. Differential Privacy

Add noise to model outputs or gradients during training:
- ε-differential privacy: P(M(D) ∈ S) ≤ e^ε * P(M(D') ∈ S)
- Where D and D' differ by one sample

### 2. Regularization

- L2 regularization: L = L₀ + λ∑w²
- Dropout: randomly zero out neurons during training
- Early stopping: prevent overfitting to training data

### 3. Model Distillation

- Train teacher model with privacy guarantees
- Distill to student model
- Reduces membership information leakage

## References

1. Shokri, R., et al. (2017). Membership Inference Attacks against Machine Learning Models
2. Salem, A., et al. (2019). ML-Leaks: Model and Data Independent Membership Inference Attacks
3. Nasr, M., et al. (2019). Comprehensive Privacy Analysis of Deep Learning
4. Carlini, N., et al. (2022). Membership Inference Attacks From First Principles