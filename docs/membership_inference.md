# Membership Inference Attacks

This document explains the membership inference attacks implemented in this project and how they can be used to evaluate the privacy of machine learning models.

## Overview

Membership inference attacks aim to determine whether a given data point was used to train a machine learning model. These attacks can reveal sensitive information about the training data and are a key privacy concern in machine learning.

## Attack Types

### 1. Shadow Model Attack

**Algorithm:**
1. Train shadow models on synthetic data
2. Use shadow models to generate training/non-training predictions
3. Train attack model to distinguish between predictions
4. Apply attack model to target model

**Implementation:**
```python
class ShadowModelAttack:
    def __init__(self, target_model, shadow_models, attack_model):
        self.target_model = target_model
        self.shadow_models = shadow_models
        self.attack_model = attack_model
    
    def train_attack_model(self, shadow_data):
        # Train attack model using shadow model predictions
        pass
    
    def infer_membership(self, target_data):
        # Use attack model to infer membership
        pass
```

### 2. Threshold Attack

**Algorithm:**
1. Compute model's confidence on target data
2. Compare confidence to threshold
3. Classify as member if confidence > threshold

**Implementation:**
```python
class ThresholdAttack:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
    
    def infer_membership(self, data):
        predictions = self.model(data)
        confidences = torch.max(predictions, dim=1)[0]
        return confidences > self.threshold
```

### 3. Loss-Based Attack

**Algorithm:**
1. Compute loss on target data
2. Compare loss to threshold
3. Classify as member if loss < threshold

**Implementation:**
```python
class LossBasedAttack:
    def __init__(self, model, criterion, threshold):
        self.model = model
        self.criterion = criterion
        self.threshold = threshold
    
    def infer_membership(self, data, labels):
        loss = self.criterion(self.model(data), labels)
        return loss < self.threshold
```

## Attack Evaluation

### 1. Metrics

- **Accuracy:** Overall attack success rate
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1 Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve

### 2. Baseline Comparison

Compare attack performance against:
- Random guessing (0.5 accuracy)
- Always predict member/non-member
- Previous attack implementations

### 3. Statistical Significance

- Use statistical tests (e.g., McNemar's test)
- Report confidence intervals
- Consider multiple runs

## Defense Mechanisms

### 1. Differential Privacy

- Add noise to gradients (DP-SGD)
- Clip gradients
- Use privacy accounting

### 2. Regularization

- L2 regularization
- Dropout
- Early stopping

### 3. Model Modifications

- Reduce model capacity
- Use ensemble methods
- Apply knowledge distillation

## Implementation Details

### 1. Data Preparation

```python
def prepare_attack_data(model, train_data, test_data):
    """Prepare data for membership inference attack."""
    # Get model predictions
    train_preds = model(train_data)
    test_preds = model(test_data)
    
    # Create labels
    train_labels = torch.ones(len(train_data))
    test_labels = torch.zeros(len(test_data))
    
    return train_preds, test_preds, train_labels, test_labels
```

### 2. Attack Model Training

```python
def train_attack_model(attack_model, train_data, train_labels):
    """Train the membership inference attack model."""
    optimizer = torch.optim.Adam(attack_model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = attack_model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
```

### 3. Evaluation

```python
def evaluate_attack(attack_model, test_data, test_labels):
    """Evaluate attack model performance."""
    with torch.no_grad():
        predictions = attack_model(test_data)
        accuracy = (predictions > 0.5).float().mean()
        precision = precision_score(test_labels, predictions > 0.5)
        recall = recall_score(test_labels, predictions > 0.5)
        f1 = f1_score(test_labels, predictions > 0.5)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

## Best Practices

1. **Attack Implementation:**
   - Use appropriate attack type for model
   - Consider computational constraints
   - Implement proper evaluation metrics

2. **Data Handling:**
   - Ensure balanced train/test split
   - Use appropriate data preprocessing
   - Consider data augmentation

3. **Model Selection:**
   - Choose attack model based on target model
   - Consider model complexity
   - Use appropriate hyperparameters

4. **Evaluation:**
   - Use multiple metrics
   - Compare against baselines
   - Report statistical significance
   - Consider multiple runs

## Command Line Interface

The project provides a command-line interface for running membership inference attacks:

```bash
python membership_inference_attacks.py \
    --attack-type shadow \
    --target-model path/to/model \
    --train-data path/to/train_data \
    --test-data path/to/test_data \
    --output-dir results/ \
    --batch-size 128 \
    --num-shadow-models 5 \
    --attack-epochs 10
```

### Arguments:

- `--attack-type`: Type of attack to run (shadow, threshold, loss)
- `--target-model`: Path to target model checkpoint
- `--train-data`: Path to training data
- `--test-data`: Path to test data
- `--output-dir`: Directory to save results
- `--batch-size`: Batch size for attack
- `--num-shadow-models`: Number of shadow models (for shadow attack)
- `--attack-epochs`: Number of epochs to train attack model

## References

1. Shokri, R., et al. (2017). Membership Inference Attacks against Machine Learning Models
2. Salem, A., et al. (2018). ML-Leaks: Model and Data Independent Membership Inference Attacks
3. Nasr, M., et al. (2019). Comprehensive Privacy Analysis of Deep Learning
4. Yeom, S., et al. (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting
5. Carlini, N., et al. (2021). Membership Inference Attacks From First Principles 