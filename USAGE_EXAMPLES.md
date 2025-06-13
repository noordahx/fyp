# Usage Examples for Differential Privacy and Membership Inference Attacks

This document provides comprehensive examples for using the differential privacy training and membership inference attack tools.

## Quick Start

### 1. Train a Standard Model (Baseline)

```bash
python scripts/train_with_dp.py \
    --method standard \
    --dataset cifar10 \
    --epochs 10 \
    --output-dir ./results/baseline
```

### 2. Train with Differential Privacy (DP-SGD)

```bash
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --epochs 10 \
    --epsilon 1.0 \
    --delta 1e-5 \
    --max-grad-norm 1.0 \
    --noise-multiplier 1.0 \
    --output-dir ./results/dp_sgd
```

### 3. Train with Output Perturbation

```bash
python scripts/train_with_dp.py \
    --method output_perturbation \
    --dataset cifar10 \
    --epochs 10 \
    --epsilon 1.0 \
    --delta 1e-5 \
    --output-dir ./results/output_pert
```

### 4. Run Membership Inference Attacks

```bash
python scripts/simple_mia_runner.py \
    --model-path ./results/baseline/cifar10_standard_eps1.0.pt \
    --dataset cifar10 \
    --attack-type all \
    --output-dir ./results/attacks
```

### 5. Run Specific Attack Types

```bash
# Threshold attack only
python scripts/simple_mia_runner.py \
    --model-path ./results/dp_sgd/cifar10_dp_sgd_custom_eps1.0.pt \
    --dataset cifar10 \
    --attack-type threshold \
    --output-dir ./results/attacks

# Loss-based attack only
python scripts/simple_mia_runner.py \
    --model-path ./results/dp_sgd/cifar10_dp_sgd_custom_eps1.0.pt \
    --dataset cifar10 \
    --attack-type loss \
    --output-dir ./results/attacks

# Shadow attack only
python scripts/simple_mia_runner.py \
    --model-path ./results/dp_sgd/cifar10_dp_sgd_custom_eps1.0.pt \
    --dataset cifar10 \
    --attack-type shadow \
    --output-dir ./results/attacks
```

## Detailed Examples

### Training Different DP Methods

#### 1. Standard Training (No Privacy)
```bash
python scripts/train_with_dp.py \
    --method standard \
    --dataset cifar10 \
    --epochs 20 \
    --lr 0.01 \
    --batch-size 128 \
    --output-dir ./results/standard
```

#### 2. DP-SGD with Different Privacy Budgets
```bash
# High privacy (low epsilon)
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --epochs 20 \
    --epsilon 0.5 \
    --delta 1e-5 \
    --max-grad-norm 1.0 \
    --noise-multiplier 2.0 \
    --output-dir ./results/dp_high_privacy

# Medium privacy
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --epochs 20 \
    --epsilon 1.0 \
    --delta 1e-5 \
    --max-grad-norm 1.0 \
    --noise-multiplier 1.0 \
    --output-dir ./results/dp_medium_privacy

# Low privacy (high epsilon)
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --epochs 20 \
    --epsilon 5.0 \
    --delta 1e-5 \
    --max-grad-norm 1.0 \
    --noise-multiplier 0.5 \
    --output-dir ./results/dp_low_privacy
```

#### 3. Output Perturbation
```bash
python scripts/train_with_dp.py \
    --method output_perturbation \
    --dataset cifar10 \
    --epochs 20 \
    --epsilon 1.0 \
    --delta 1e-5 \
    --output-dir ./results/output_perturbation
```

### Running Comprehensive Attack Evaluation

#### 1. Train Multiple Models and Run Attacks
```bash
# Train standard model
python scripts/train_with_dp.py \
    --method standard \
    --dataset cifar10 \
    --epochs 10 \
    --output-dir ./results/comparison \
    --model-name standard_model

# Train DP model
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --epochs 10 \
    --epsilon 1.0 \
    --output-dir ./results/comparison \
    --model-name dp_model

# Run attacks on standard model
python scripts/simple_mia_runner.py \
    --model-path ./results/comparison/standard_model.pt \
    --dataset cifar10 \
    --attack-type all \
    --output-dir ./results/comparison/standard_attacks

# Run attacks on DP model
python scripts/simple_mia_runner.py \
    --model-path ./results/comparison/dp_model.pt \
    --dataset cifar10 \
    --attack-type all \
    --output-dir ./results/comparison/dp_attacks
```

#### 2. Automated Training with Attack Evaluation
```bash
# Train and immediately run attacks
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --epochs 10 \
    --epsilon 1.0 \
    --output-dir ./results/auto_eval \
    --run-attacks
```

### Different Datasets

#### MNIST
```bash
# Train on MNIST
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset mnist \
    --epochs 10 \
    --epsilon 1.0 \
    --output-dir ./results/mnist

# Attack MNIST model
python scripts/simple_mia_runner.py \
    --model-path ./results/mnist/mnist_dp_sgd_custom_eps1.0.pt \
    --dataset mnist \
    --attack-type all \
    --output-dir ./results/mnist_attacks
```

#### CIFAR-100
```bash
# Train on CIFAR-100
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar100 \
    --epochs 15 \
    --epsilon 1.0 \
    --output-dir ./results/cifar100

# Attack CIFAR-100 model
python scripts/simple_mia_runner.py \
    --model-path ./results/cifar100/cifar100_dp_sgd_custom_eps1.0.pt \
    --dataset cifar100 \
    --attack-type all \
    --output-dir ./results/cifar100_attacks
```

## Advanced Usage

### Custom Model Names and Paths
```bash
# Custom model name
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --epochs 10 \
    --epsilon 2.0 \
    --model-name my_custom_model \
    --output-dir ./results/custom

# Use custom model path for attacks
python scripts/simple_mia_runner.py \
    --model-path ./results/custom/my_custom_model.pt \
    --dataset cifar10 \
    --attack-type threshold \
    --output-dir ./results/custom_attacks
```

### Hyperparameter Tuning
```bash
# Different learning rates
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --lr 0.001 \
    --epochs 15 \
    --epsilon 1.0 \
    --model-name low_lr_model \
    --output-dir ./results/lr_tuning

# Different batch sizes
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --batch-size 64 \
    --epochs 15 \
    --epsilon 1.0 \
    --model-name small_batch_model \
    --output-dir ./results/batch_tuning
```

### GPU Training
```bash
# Force GPU usage
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --epochs 20 \
    --epsilon 1.0 \
    --device cuda \
    --output-dir ./results/gpu_training

# Force CPU usage
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --epochs 10 \
    --epsilon 1.0 \
    --device cpu \
    --output-dir ./results/cpu_training
```

## Privacy-Utility Trade-off Analysis

### Epsilon Sweep
```bash
# Train models with different epsilon values
for eps in 0.1 0.5 1.0 2.0 5.0 10.0; do
    python scripts/train_with_dp.py \
        --method dp_sgd_custom \
        --dataset cifar10 \
        --epochs 10 \
        --epsilon $eps \
        --model-name "eps_${eps}_model" \
        --output-dir ./results/epsilon_sweep
done

# Run attacks on all models
for eps in 0.1 0.5 1.0 2.0 5.0 10.0; do
    python scripts/simple_mia_runner.py \
        --model-path "./results/epsilon_sweep/eps_${eps}_model.pt" \
        --dataset cifar10 \
        --attack-type all \
        --output-dir "./results/epsilon_sweep/attacks_eps_${eps}"
done
```

### Noise Multiplier Sweep
```bash
# Train models with different noise multipliers
for noise in 0.5 1.0 1.5 2.0; do
    python scripts/train_with_dp.py \
        --method dp_sgd_custom \
        --dataset cifar10 \
        --epochs 10 \
        --epsilon 1.0 \
        --noise-multiplier $noise \
        --model-name "noise_${noise}_model" \
        --output-dir ./results/noise_sweep
done
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA out of memory**: Reduce batch size
   ```bash
   python scripts/train_with_dp.py \
       --method dp_sgd_custom \
       --dataset cifar10 \
       --batch-size 64 \
       --epochs 10
   ```

2. **Slow training**: Reduce epochs or use CPU
   ```bash
   python scripts/train_with_dp.py \
       --method dp_sgd_custom \
       --dataset cifar10 \
       --epochs 5 \
       --device cpu
   ```

3. **Attack failures**: Check model path and dataset match
   ```bash
   # Make sure dataset matches the one used for training
   python scripts/simple_mia_runner.py \
       --model-path ./results/model.pt \
       --dataset cifar10 \
       --attack-type threshold
   ```

### Performance Tips

1. **Use GPU for faster training**:
   ```bash
   python scripts/train_with_dp.py --device cuda
   ```

2. **Reduce shadow models for faster attacks**:
   - Shadow attacks are computationally expensive
   - Consider using threshold or loss attacks for quick evaluation

3. **Start with small epochs for testing**:
   ```bash
   python scripts/train_with_dp.py --epochs 3
   ```

## Output Files

### Training Outputs
- `model_name.pt`: Trained model weights
- `model_name_history.json`: Training history and metadata
- `model_name_summary.json`: Training summary

### Attack Outputs
- `attack_results.json`: Detailed attack results
- Attack-specific metrics and predictions

## Next Steps

1. **Analyze Results**: Compare attack success rates between standard and DP models
2. **Tune Parameters**: Adjust epsilon, delta, and noise multipliers based on results
3. **Scale Up**: Run larger experiments with more epochs and datasets
4. **Visualize**: Create plots showing privacy-utility trade-offs 