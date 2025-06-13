# Shadow Models Training Documentation

## Overview

The shadow models training script (`scripts/train_shadow_models.py`) implements the shadow model attack methodology for membership inference attacks. Shadow models are trained to mimic the target model's behavior and generate training data for the final attack model by providing known member/non-member examples.

## Purpose

Shadow models serve several critical functions in membership inference attacks:

1. **Attack Data Generation**: Create labeled training data (member/non-member) for the attack model
2. **Target Model Simulation**: Mimic the target model's training process and behavior
3. **Feature Extraction**: Generate prediction confidence features that correlate with membership
4. **Attack Generalization**: Ensure attack model generalizes beyond specific target model characteristics

## Mathematical Foundation

### Shadow Model Attack Theory

The shadow model attack, introduced by Shokri et al. (2017), is based on the principle that models trained on different datasets but with similar distributions will exhibit similar membership leakage patterns.

**Core Assumption:**
```
If M_shadow(D_shadow) ≈ M_target(D_target) in distribution,
then membership patterns in M_shadow generalize to M_target
```

### Attack Data Generation Process

For each shadow model `M_i` trained on dataset `D_i`:

1. **Member Examples**: `{(f(x), 1) | x ∈ D_i}`
2. **Non-member Examples**: `{(f(x), 0) | x ∉ D_i}`

Where `f(x)` represents the feature vector extracted from model outputs (typically confidence scores).

### Anti-Overfitting Mathematical Framework

Overfitting increases membership signal, making attacks easier. We implement several regularization techniques:

#### 1. L2 Regularization Enhancement
```
L_regularized(θ) = L_original(θ) + λ_enhanced ||θ||²₂
```
Where `λ_enhanced = 2 × λ_original`

#### 2. Dropout Regularization
```
p_dropout_enhanced = min(0.6, p_original + 0.2)
```

#### 3. Early Stopping Criterion
```
Stop training when: val_loss(epoch) > val_loss(epoch - patience)
```
With reduced patience: `patience_enhanced = min(5, patience_original)`

#### 4. Learning Rate Reduction
```
lr_enhanced = 0.7 × lr_original
```

## Implementation Details

### Class Structure

```python
class ShadowModelTrainer:
    """
    Shadow Models Trainer with Anti-Overfitting Measures
    """
```

### Key Methods

#### 1. `setup_anti_overfitting_config()`
Configures enhanced regularization parameters:

```python
# Reduce epochs to prevent overfitting
self.shadow_cfg.target_model.epochs = max(15, original_epochs // 3)

# Increase dropout rate
self.shadow_cfg.target_model.dropout_rate = min(0.6, original_dropout + 0.2)

# Increase weight decay (L2 regularization)
self.shadow_cfg.target_model.weight_decay = min(0.1, original_weight_decay * 2)
```

#### 2. `create_shadow_data_splits()`
Creates non-overlapping data splits for each shadow model:

```python
def create_shadow_data_splits(self, shadow_idx):
    total_train_indices = np.arange(len(self.trainset))
    
    # Calculate split sizes
    train_size = self.CFG_ATTACK.shadow.shadow_train_size
    eval_size = self.CFG_ATTACK.shadow.shadow_eval_size
    test_size = self.CFG_ATTACK.shadow.shadow_test_size
    
    # Create non-overlapping splits
    start_idx = (shadow_idx * required_size) % len(total_train_indices)
    train_indices = total_train_indices[start_idx:start_idx + train_size]
    # ... (eval and test indices)
    
    return train_indices, eval_indices, test_indices
```

#### 3. `evaluate_shadow_model()`
Evaluates overfitting indicators:

```python
# Calculate overfitting indicators
accuracy_gap = train_metrics['accuracy'] - eval_metrics['accuracy']
confidence_gap = train_metrics['mean_confidence'] - eval_metrics['mean_confidence']

# Flag overfitting
is_overfitting = accuracy_gap > 0.1 or confidence_gap > 0.05
```

#### 4. `create_attack_dataset()`
Generates attack training data from shadow models:

```python
for shadow_model, train_indices, eval_indices, test_indices in shadow_models_info:
    # Member data (from training set)
    member_features = extract_features(shadow_model, train_data)
    member_labels = np.ones(len(member_features))
    
    # Non-member data (from test set)
    non_member_features = extract_features(shadow_model, test_data)
    non_member_labels = np.zeros(len(non_member_features))
    
    # Combine into attack dataset
    attack_data = combine(member_features, non_member_features, 
                         member_labels, non_member_labels)
```

## Configuration Parameters

### Shadow Model Parameters

```yaml
shadow:
  num_shadow_models: 32          # Number of shadow models
  shadow_train_size: 3000        # Training samples per shadow model
  shadow_eval_size: 3000         # Evaluation samples per shadow model  
  shadow_test_size: 3000         # Test samples per shadow model
```

### Anti-Overfitting Configuration

The script automatically modifies these parameters:

```yaml
# Original → Enhanced
epochs: 100 → 33                # Reduced to 1/3 of original
dropout_rate: 0.4 → 0.6         # Increased by 0.2
weight_decay: 0.01 → 0.02       # Doubled
learning_rate: 0.0005 → 0.00035 # Reduced by 30%
early_stop_patience: 10 → 5     # More aggressive early stopping
```

### Attack Dataset Configuration

```yaml
attack:
  output_dim: 10                 # Number of classes (feature dimension)
  catboost:
    iterations: 200              # CatBoost training iterations
    depth: 2                     # Tree depth
    learning_rate: 0.25          # CatBoost learning rate
    loss_function: "Logloss"     # Binary classification loss
```

## Usage Instructions

### Basic Usage

```bash
cd fyp
python scripts/train_shadow_models.py
```

### Programmatic Usage

```python
from scripts.train_shadow_models import ShadowModelTrainer

# Initialize trainer
trainer = ShadowModelTrainer("configs/mia_config.yaml")

# Run complete pipeline
shadow_models_info, attack_model, attack_dataset = trainer.run_complete_pipeline()

# Access results
print(f"Trained {len(shadow_models_info)} shadow models")
print(f"Attack dataset size: {len(attack_dataset)}")
```

### Advanced Configuration

```python
# Custom anti-overfitting parameters
trainer = ShadowModelTrainer()
trainer.shadow_cfg.target_model.epochs = 20
trainer.shadow_cfg.target_model.dropout_rate = 0.7
trainer.shadow_cfg.target_model.weight_decay = 0.05

# Run training
results = trainer.run_complete_pipeline()
```

## Evaluation Metrics

### Shadow Model Performance Metrics

#### 1. Individual Model Metrics
- **Training Accuracy**: Performance on training data (members)
- **Evaluation Accuracy**: Performance on held-out data (non-members)
- **Training Loss**: Cross-entropy loss on training data
- **Evaluation Loss**: Cross-entropy loss on evaluation data

#### 2. Overfitting Indicators
- **Accuracy Gap**: `train_accuracy - eval_accuracy`
- **Confidence Gap**: `mean_train_confidence - mean_eval_confidence`
- **Overfitting Flag**: Boolean indicator based on thresholds

```python
# Overfitting detection
is_overfitting = (accuracy_gap > 0.1) or (confidence_gap > 0.05)
```

#### 3. Aggregate Metrics
- **Mean Performance**: Average across all shadow models
- **Performance Variance**: Consistency across models
- **Overfitting Rate**: Percentage of models showing overfitting

### Attack Dataset Quality Metrics

#### 1. Dataset Balance
```python
balance_ratio = num_members / (num_members + num_non_members)
# Ideal: 0.5 (perfectly balanced)
```

#### 2. Feature Distribution
- **Member Feature Statistics**: Mean, std, percentiles
- **Non-member Feature Statistics**: Mean, std, percentiles
- **Separability**: Statistical distance between distributions

#### 3. Dataset Size
- **Total Samples**: Combined from all shadow models
- **Samples per Shadow Model**: Consistency check
- **Feature Dimensionality**: Number of output classes

## Output Files

### Directory Structure

```
models/shadow_model/
├── shadow_model_0.pth         # Individual shadow model weights
├── shadow_model_1.pth
├── ...
├── shadow_model_N.pth
├── shadow_models_metrics.npz   # Performance metrics
├── attack_dataset.csv         # Attack training dataset
├── attack_model.cbm           # Trained CatBoost attack model
└── shadow_models_analysis.png # Analysis visualization
```

### Metrics File Structure

```python
# shadow_models_metrics.npz
{
    'shadow_idx': [0, 1, 2, ...],
    'train_accuracy': [0.85, 0.87, ...],
    'eval_accuracy': [0.82, 0.84, ...],
    'accuracy_gap': [0.03, 0.03, ...],
    'confidence_gap': [0.02, 0.01, ...],
    'is_overfitting': [False, False, ...]
}
```

### Attack Dataset Structure

```csv
# attack_dataset.csv
confidence_0, confidence_1, ..., confidence_9, label, shadow_model_id
0.1, 0.05, ..., 0.8, 1, 0    # Member example from shadow model 0
0.3, 0.2, ..., 0.1, 0, 0     # Non-member example from shadow model 0
...
```

## Visualization

The script generates comprehensive analysis plots:

### 1. Training vs Evaluation Accuracy Scatter Plot
Shows correlation between training and evaluation performance with overfitting detection.

### 2. Accuracy Gap Distribution
Histogram showing distribution of overfitting across shadow models.

### 3. Confidence Gap Distribution  
Distribution of confidence differences between training and evaluation data.

### 4. Overfitting Analysis Bar Chart
Count of models showing normal vs overfitting behavior.

### 5. Individual Model Performance Lines
Performance trends across all shadow models.

### 6. Summary Statistics Panel
Text summary of key metrics and anti-overfitting effectiveness.

## Best Practices

### 1. Shadow Model Count Selection

**Recommended ranges:**
- **Minimum**: 10 models (for basic attacks)
- **Standard**: 32 models (good balance)
- **High-quality**: 64+ models (better attack performance)

**Trade-offs:**
```
More shadow models → Better attack quality + Higher computational cost
Fewer shadow models → Faster training + Lower attack quality
```

### 2. Data Split Strategy

**Balanced splits:**
```python
# Ensure similar split sizes
train_size = eval_size = test_size = 3000  # Equal splits
```

**Overlap prevention:**
```python
# Use modular arithmetic for non-overlapping data
start_idx = (shadow_idx * total_size) % dataset_size
```

### 3. Anti-Overfitting Configuration

**Conservative approach** (stronger privacy):
```yaml
epochs: 15
dropout_rate: 0.7
weight_decay: 0.1
noise_multiplier: 2.0
```

**Moderate approach** (balanced):
```yaml
epochs: 25
dropout_rate: 0.5
weight_decay: 0.05
noise_multiplier: 1.5
```

### 4. Quality Control

**Monitor these indicators:**
- Overfitting rate < 20%
- Accuracy gap < 0.1
- Consistent performance across models
- Balanced attack dataset

## Troubleshooting

### Common Issues

#### 1. Excessive Overfitting
**Problem:** >50% of shadow models show overfitting
**Solutions:**
```python
# Strengthen regularization
dropout_rate = 0.8
weight_decay = 0.2
epochs = 10

# Add noise
noise_multiplier = 3.0
```

#### 2. Poor Shadow Model Performance
**Problem:** Low accuracy across all shadow models
**Solutions:**
```python
# Increase training epochs
epochs = 30

# Reduce regularization slightly
dropout_rate = 0.4
weight_decay = 0.01

# Check data quality and preprocessing
```

#### 3. Memory Issues
**Problem:** CUDA out of memory with multiple models
**Solutions:**
```python
# Reduce batch size
train_batch_size = 64
eval_batch_size = 128

# Train models sequentially and move to CPU
model.cpu()
torch.cuda.empty_cache()
```

#### 4. Imbalanced Attack Dataset
**Problem:** Unequal member/non-member samples
**Solutions:**
```python
# Ensure equal split sizes
shadow_train_size = shadow_test_size = 3000

# Check data splitting logic
# Implement balanced sampling
```

### Performance Optimization

#### 1. Training Speed
```python
# Parallel data loading
num_workers = 8
pin_memory = True

# Mixed precision training
use_amp = True

# Reduce model complexity for shadows
hidden_dim = 256  # Smaller than target
```

#### 2. Memory Efficiency
```python
# Gradient accumulation
accumulation_steps = 4

# Model checkpointing
del shadow_model  # After training
gc.collect()
torch.cuda.empty_cache()
```

## Advanced Features

### 1. Adaptive Anti-Overfitting

```python
def adaptive_regularization(train_acc, val_acc):
    """Adjust regularization based on overfitting level"""
    gap = train_acc - val_acc
    if gap > 0.15:
        return {
            'dropout_rate': 0.8,
            'weight_decay': 0.1,
            'lr_factor': 0.5
        }
    elif gap > 0.1:
        return {
            'dropout_rate': 0.6,
            'weight_decay': 0.05,
            'lr_factor': 0.7
        }
    else:
        return {}  # No additional regularization
```

### 2. Shadow Model Diversity

```python
# Different architectures for different shadow models
architectures = ['resnet18', 'resnet34', 'densenet121']
shadow_arch = architectures[shadow_idx % len(architectures)]
```

### 3. Dynamic Data Splits

```python
# Adaptive split sizes based on available data
def adaptive_split_size(dataset_size, num_models):
    optimal_size = dataset_size // (num_models * 3)  # train, eval, test
    return min(optimal_size, 5000)  # Cap at reasonable size
```

## Integration with Attack Pipeline

### Data Flow

```
Original Dataset → Shadow Data Splits → Shadow Models → Attack Dataset → Attack Model
      ↓                   ↓                  ↓              ↓             ↓
  Train/Test Split   Non-overlapping    Member/Non-member  Feature       Final Attack
                        Subsets          Examples          Extraction     Classifier
```

### Interface with Attack Evaluation

```python
# Output format for attack model training
attack_features = {
    'features': np.array,      # Shape: (n_samples, n_classes)
    'labels': np.array,        # Shape: (n_samples,) - 0/1 for non-member/member
    'shadow_id': np.array      # Shape: (n_samples,) - which shadow model
}
```

## Security Considerations

### 1. Information Isolation
- Each shadow model trained on independent data
- No information sharing between shadow models
- Separate random seeds for reproducibility

### 2. Realistic Attack Assumptions
- Shadow models use different data than target
- Limited knowledge of target model architecture
- Practical computational constraints

### 3. Privacy-Preserving Evaluation
- Avoid information leakage during evaluation
- Use held-out test sets for final attack assessment
- Implement proper cross-validation procedures

## Future Extensions

### 1. Federated Shadow Models
Adaptation for federated learning scenarios where shadow models are distributed.

### 2. Transfer Learning Shadows
Using pre-trained models as shadow model initialization.

### 3. Adversarial Shadow Training
Training shadow models to be robust against defenses.

### 4. Multi-Modal Shadow Models
Extension to handle different data modalities (text, images, audio).

## References

1. Shokri, R., et al. "Membership inference attacks against machine learning models." 2017 IEEE Symposium on Security and Privacy (SP).

2. Salem, A., et al. "ML-Leaks: Model and data independent membership inference attacks and defenses on machine learning models." NDSS 2019.

3. Song, L., & Mittal, P. "Systematic evaluation of privacy risks of machine learning models." arXiv preprint arXiv:2003.10595 (2020).

4. Nasr, M., et al. "Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning." 2019 IEEE symposium on security and privacy (SP).