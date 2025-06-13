# Differential Privacy in Machine Learning

This document explains the mathematical foundations of differential privacy (DP) methods implemented in this project for protecting machine learning models against membership inference attacks.

## Overview

Differential privacy provides a rigorous mathematical framework for privacy-preserving machine learning. It ensures that the presence or absence of any individual training example has a limited impact on the model's output.

## Mathematical Foundations

### 1. ε-Differential Privacy

A randomized algorithm M is ε-differentially private if for all datasets D and D' differing in one sample, and all possible outputs S:

P(M(D) ∈ S) ≤ e^ε * P(M(D') ∈ S)

Where:
- ε is the privacy budget (smaller ε = stronger privacy)
- D and D' are neighboring datasets
- M is the algorithm (e.g., training procedure)
- S is any subset of possible outputs

### 2. (ε, δ)-Differential Privacy

A relaxation of pure differential privacy:

P(M(D) ∈ S) ≤ e^ε * P(M(D') ∈ S) + δ

Where δ represents a small probability of privacy violation.

## Implementation Methods

### 1. DP-SGD (Differential Private Stochastic Gradient Descent)

**Algorithm:**
1. For each batch:
   - Compute gradients g
   - Clip gradients: ḡ = g / max(1, ||g||₂/C)
   - Add noise: g̃ = ḡ + N(0, σ²C²I)
   - Update weights: w = w - ηg̃

**Privacy Analysis:**
- Noise scale: σ = √(2log(1.25/δ)) * (√T/ε)
- Where T is number of training steps
- Privacy budget: ε = O(√T * log(1/δ))

### 2. PATE (Private Aggregation of Teacher Ensembles)

**Algorithm:**
1. Train K teacher models on disjoint data
2. For each query x:
   - Get teacher predictions: p₁, p₂, ..., pₖ
   - Add noise to vote counts: ñᵢ = nᵢ + Lap(1/ε)
   - Output: argmaxᵢ ñᵢ

**Privacy Guarantees:**
- ε-teacher privacy: εₜ = O(√K * log(1/δ))
- ε-student privacy: εₛ = O(√N * log(1/δ))
- Where N is number of student queries

### 3. DP-FedAvg (Differential Private Federated Averaging)

**Algorithm:**
1. For each client:
   - Train local model
   - Clip updates: Δ̄ = Δ / max(1, ||Δ||₂/C)
2. Server:
   - Aggregate updates: Δ̃ = (1/K)∑Δ̄ + N(0, σ²C²I)
   - Update global model: w = w - ηΔ̃

**Privacy Analysis:**
- Per-round privacy: εᵣ = O(√K * log(1/δ))
- Total privacy: ε = O(√R * εᵣ)
- Where R is number of rounds

## Privacy Budget Management

### 1. Composition Theorems

**Sequential Composition:**
- For k ε-DP algorithms: total privacy = kε
- For k (ε,δ)-DP algorithms: total privacy = (kε, kδ)

**Parallel Composition:**
- For disjoint datasets: privacy = max(εᵢ)
- For overlapping datasets: privacy = sum(εᵢ)

### 2. Privacy Accounting

**Moments Accountant:**
- Tracks privacy loss across training
- Provides tighter bounds than composition
- Supports adaptive noise levels

**Rényi Differential Privacy:**
- Generalizes (ε,δ)-DP
- Better composition properties
- Easier to analyze complex algorithms

## Implementation Details

### 1. Gradient Clipping

```python
def clip_gradients(gradients, clip_norm):
    """Clip gradients to have maximum L2 norm."""
    total_norm = torch.norm(torch.stack([g.norm() for g in gradients]))
    clip_coef = clip_norm / (total_norm + 1e-6)
    return [g * clip_coef for g in gradients]
```

### 2. Noise Addition

```python
def add_noise(gradients, noise_multiplier, clip_norm):
    """Add Gaussian noise to gradients."""
    noise = torch.randn_like(gradients) * noise_multiplier * clip_norm
    return gradients + noise
```

### 3. Privacy Accounting

```python
def compute_privacy_budget(steps, noise_multiplier, delta):
    """Compute privacy budget using moments accountant."""
    # Implementation of moments accountant
    pass
```

## Trade-offs and Considerations

### 1. Privacy vs. Utility

- Smaller ε = stronger privacy but lower accuracy
- Larger δ = weaker privacy but better utility
- Need to balance based on use case

### 2. Computational Overhead

- Gradient clipping: O(d) where d is model dimension
- Noise addition: O(d) per batch
- Privacy accounting: O(1) per step

### 3. Hyperparameter Tuning

- Clip norm C: affects gradient scale
- Noise multiplier σ: affects privacy budget
- Learning rate η: needs adjustment for noisy gradients

## Best Practices

1. **Privacy Budget Management:**
   - Start with conservative ε
   - Monitor privacy loss during training
   - Use privacy accounting tools

2. **Model Architecture:**
   - Simpler models often work better with DP
   - Consider model capacity vs. privacy budget
   - Use appropriate regularization

3. **Training Process:**
   - Larger batch sizes help with privacy
   - Careful learning rate scheduling
   - Early stopping based on validation

## References

1. Abadi, M., et al. (2016). Deep Learning with Differential Privacy
2. Papernot, N., et al. (2017). Semi-supervised Knowledge Transfer for Deep Learning
3. McMahan, B., et al. (2018). A General Approach to Adding Differential Privacy
4. Mironov, I. (2017). Rényi Differential Privacy
5. Dwork, C., et al. (2014). The Algorithmic Foundations of Differential Privacy 