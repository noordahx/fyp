# Target Model Training Documentation

## Overview

The target model training script (`scripts/train_target_model.py`) is responsible for training the primary model that will be subjected to membership inference attacks. This model serves as the "victim" model in the experimental setup, where attackers attempt to determine whether specific data points were used during training.

## Purpose

The target model is the central component of membership inference attack (MIA) experiments. It:

1. **Simulates a real-world ML model** that an attacker might target
2. **Creates realistic training scenarios** by using subset of test data for training
3. **Provides ground truth labels** for membership (member/non-member) evaluation
4. **Establishes baseline performance** for comparison with privacy-protected models

## Mathematical Foundation

### Membership Inference Problem Setup

Given a machine learning model `M` trained on dataset `D_train`, and a data point `x`, the membership inference problem asks:

```
Is x ∈ D_train?
```

The target model training creates this scenario by:
- Training model `M` on subset `S ⊂ D_test` (members)
- Keeping remaining data `D_test \ S` as non-members
- This provides ground truth for membership labels

### Model Architecture

The default architecture uses ResNet-18 with modifications for CIFAR-10:

```
Input: (3, 32, 32) → ResNet-18 → FC(512, 10) → Softmax → Output: (10,)
```

### Training Objective

Standard cross-entropy loss with L2 regularization:

```
L(θ) = -∑_{i=1}^N log P(y_i | x_i, θ) + λ||θ||²₂
```

Where:
- `θ`: Model parameters
- `λ`: Weight decay coefficient
- `N`: Number of training samples

## Implementation Details

### Class Structure

```python
class TargetModelTrainer:
    """
    Comprehensive target model trainer with evaluation metrics
    """
```

### Key Methods

#### 1. `load_data()`
Loads CIFAR-10 dataset and creates data loaders.

**Returns:**
- `trainset`, `testset`: PyTorch datasets
- `trainloader`, `testloader`: Data loaders

#### 2. `create_target_subsets()`
Creates member and non-member data splits.

**Logic:**
```python
total_test_indices = np.arange(len(testset))
target_train_indices = np.random.choice(
    total_test_indices, 
    size=CFG.target_model.target_train_size, 
    replace=False
)
target_eval_indices = np.setdiff1d(total_test_indices, target_train_indices)
```

#### 3. `evaluate_model_comprehensive()`
Performs comprehensive evaluation with multiple metrics.

**Evaluation Datasets:**
- Full test set (overall performance)
- Member data (training subset)
- Non-member data (held-out subset)

## Configuration Parameters

### Training Parameters

```yaml
target_model:
  architecture: resnet18          # Model architecture
  epochs: 100                     # Training epochs
  learning_rate: 0.0005          # Initial learning rate
  learning_rate_decay: 0.00001   # Learning rate decay
  weight_decay: 0.01             # L2 regularization
  dropout_rate: 0.4              # Dropout probability
  target_train_size: 7500        # Number of member samples
  train_batch_size: 128          # Training batch size
  eval_batch_size: 256           # Evaluation batch size
```

### Data Augmentation

```yaml
use_data_augmentation: true
```

Applies:
- Random horizontal flips
- Random crops with padding
- Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Usage Instructions

### Basic Usage

```bash
cd fyp
python scripts/train_target_model.py
```

### Programmatic Usage

```python
from scripts.train_target_model import TargetModelTrainer

# Initialize trainer
trainer = TargetModelTrainer("configs/mia_config.yaml")

# Run complete pipeline
model, metrics = trainer.run_complete_pipeline()

# Access results
print(f"Test accuracy: {metrics['test_metrics']['accuracy']:.4f}")
print(f"Member accuracy: {metrics['member_metrics']['accuracy']:.4f}")
print(f"Non-member accuracy: {metrics['non_member_metrics']['accuracy']:.4f}")
```

## Evaluation Metrics

### Classification Metrics

1. **Accuracy**: Overall classification accuracy
2. **Precision/Recall/F1**: Per-class and weighted averages
3. **ROC AUC**: Area under ROC curve (multi-class)
4. **Confusion Matrix**: Detailed classification breakdown

### Confidence Statistics

1. **Mean Confidence**: Average prediction confidence
2. **Confidence Distribution**: Statistical measures (std, min, max, percentiles)
3. **Member vs Non-member Confidence**: Comparison between groups

### Privacy-Related Metrics

1. **Confidence Gap**: Difference in confidence between members and non-members
   ```
   Gap = mean(confidence_members) - mean(confidence_non_members)
   ```

2. **Membership Advantage**: How much easier it is to identify members
   ```
   Advantage = accuracy_members - accuracy_non_members
   ```

## Output Files

### Model Files

```
models/target_model/
├── target_model.pth           # Trained model weights
├── target_indices.npz         # Member/non-member indices
├── target_metrics.npz         # Comprehensive metrics
└── target_model_evaluation.png # Evaluation plots
```

### Metrics Structure

```python
{
    'test_metrics': {
        'accuracy': float,
        'precision_weighted': float,
        'recall_weighted': float,
        'f1_weighted': float,
        'confusion_matrix': np.ndarray,
        'confidence_stats': dict
    },
    'member_metrics': { ... },      # Same structure for members
    'non_member_metrics': { ... },  # Same structure for non-members
    'config': dict                  # Training configuration
}
```

## Visualization

The script generates comprehensive evaluation plots:

### 1. Accuracy Comparison
Bar chart comparing accuracy across:
- Full test set
- Member data
- Non-member data

### 2. Confusion Matrix
Heatmap showing classification performance on full test set.

### 3. Per-Class Performance
Bar chart of F1 scores for each class.

### 4. Confidence Analysis
Comparison of confidence statistics between members and non-members.

### 5. Precision-Recall Comparison
Side-by-side comparison of precision and recall across datasets.

### 6. Configuration Summary
Text summary of model hyperparameters.

## Best Practices

### 1. Data Splitting Strategy

**Recommended approach:**
- Use test data for target training (simulates realistic attack scenario)
- Reserve separate validation set for hyperparameter tuning
- Ensure no data leakage between splits

### 2. Model Selection

**For realistic attacks:**
- Choose architectures commonly used in practice
- Avoid excessive regularization (makes attacks easier)
- Use standard training procedures

### 3. Hyperparameter Tuning

**Balance considerations:**
- Model performance (higher accuracy = more realistic)
- Attack vulnerability (some overfitting makes attacks more effective)
- Training stability (reproducible results)

## Troubleshooting

### Common Issues

#### 1. Memory Errors
**Problem:** CUDA out of memory during training
**Solution:**
```yaml
target_model:
  train_batch_size: 64  # Reduce from 128
  eval_batch_size: 128  # Reduce from 256
```

#### 2. Convergence Issues
**Problem:** Model not converging or poor performance
**Solutions:**
- Check learning rate (try 0.001 or 0.0001)
- Verify data preprocessing
- Increase epochs or adjust learning rate schedule

#### 3. Reproducibility Issues
**Problem:** Different results across runs
**Solutions:**
- Verify seed setting in config
- Check for non-deterministic operations
- Use `torch.backends.cudnn.deterministic = True`

### Configuration Issues

#### 1. Invalid Architecture
**Error:** Model architecture not found
**Solution:** Check `architecture` parameter in config matches available models

#### 2. Insufficient Data
**Error:** Not enough data for target_train_size
**Solution:** Reduce `target_train_size` or check dataset loading

### Performance Optimization

#### 1. Training Speed
```yaml
target_model:
  num_workers: 8        # Increase data loading workers
  pin_memory: true      # Enable for GPU training
```

#### 2. Evaluation Speed
```python
# Use smaller evaluation batches for memory efficiency
eval_batch_size: 512
```

## Advanced Configuration

### Custom Data Splits

```python
# Custom member/non-member split ratios
target_train_size: 5000    # Reduce for faster training
```

### Modified Training Schedule

```yaml
learning_rate_step_size: 20    # LR decay frequency
learning_rate_gamma: 0.5       # LR decay factor
early_stop_patience: 15        # Early stopping patience
```

### Enhanced Regularization

```yaml
dropout_rate: 0.5              # Increase dropout
weight_decay: 0.05             # Increase L2 regularization
```

## Integration with Attack Pipeline

The target model training integrates with other pipeline components:

1. **Shadow Model Training**: Uses same architecture and hyperparameters
2. **Attack Evaluation**: Provides member/non-member ground truth
3. **Privacy Protection**: Serves as baseline for DP-protected models

### Data Flow

```
Target Model Training → Shadow Model Training → Attack Execution → Privacy Protection
         ↓                       ↓                    ↓                    ↓
   Ground Truth Labels    Attack Training Data   Attack Results    Protected Models
```

## Security Considerations

### Information Leakage Prevention

1. **Separate environments** for target and attack model training
2. **Independent random seeds** for different components
3. **Careful data handling** to prevent unintended information sharing

### Realistic Attack Scenarios

1. **Limited attacker knowledge** - attacker shouldn't know exact training procedure
2. **Practical constraints** - consider real-world limitations
3. **Diverse attack methods** - test against multiple attack strategies

## Future Extensions

### 1. Multi-Dataset Support
Extend to support other datasets (ImageNet, MNIST, custom datasets).

### 2. Additional Architectures
Support for transformers, CNNs, and other modern architectures.

### 3. Federated Learning Scenarios
Adaptation for federated learning membership inference attacks.

### 4. Continuous Learning
Support for online/continual learning scenarios.

## References

1. Shokri, R., et al. "Membership inference attacks against machine learning models." 2017 IEEE Symposium on Security and Privacy (SP).

2. Yeom, S., et al. "Privacy risk in machine learning: Analyzing the connection to overfitting." 2018 IEEE 31st Computer Security Foundations Symposium (CSF).

3. Song, L., et al. "Systematic evaluation of privacy risks of machine learning models." arXiv preprint arXiv:2003.10595 (2020).