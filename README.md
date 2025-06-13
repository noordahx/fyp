# Membership Inference Attack and Differential Privacy Framework

A comprehensive framework for training neural networks with differential privacy and evaluating their vulnerability to membership inference attacks.

## Overview

This repository provides:
- **Differential Privacy Training**: Multiple DP methods including DP-SGD, PATE, and Output Perturbation
- **Membership Inference Attacks**: Shadow models, reference models, and leave-one-out attacks
- **Privacy Evaluation**: Tools to assess privacy risks and trade-offs

## Repository Structure

```
fyp/
├── scripts/
│   ├── train_with_dp.py      # Main DP training script
│   └── mia_runner.py         # MIA evaluation script
├── mia_lib/
│   ├── attack/               # Attack implementations
│   │   ├── attack_shadow.py
│   │   ├── attack_reference.py
│   │   └── attack_leave_one_out.py
│   ├── dp_training.py        # DP training methods
│   ├── data.py               # Dataset utilities
│   ├── models.py             # Model architectures
│   ├── trainer.py            # Base trainer
│   └── utils.py              # Utility functions
├── dp_lib/                   # DP library components
│   └── dp_methods/           # DP method implementations
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fyp
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Model with Differential Privacy

Train a standard model (baseline):
```bash
python scripts/train_with_dp.py --method standard --dataset cifar10 --epochs 20
```

Train with DP-SGD:
```bash
python scripts/train_with_dp.py --method dp_sgd_custom --dataset cifar10 --epsilon 1.0 --epochs 20
```

Train with PATE:
```bash
python scripts/train_with_dp.py --method pate --dataset cifar10 --num-teachers 5 --epsilon 8.0
```

Train with Output Perturbation:
```bash
python scripts/train_with_dp.py --method output_perturbation --dataset cifar10 --epsilon 1.0
```

### 2. Evaluate Privacy with MIA

Run shadow model attack:
```bash
python scripts/mia_runner.py --model results/cifar10_standard.pt --dataset cifar10 --attack shadow
```

Run all attacks:
```bash
python scripts/mia_runner.py --model results/cifar10_dp_sgd_custom_eps1.0.pt --dataset cifar10 --attack all
```

## Detailed Usage

### Training with Differential Privacy

The `train_with_dp.py` script supports multiple DP methods:

#### Command Line Arguments

**Dataset Options:**
- `--dataset`: Choose from `cifar10`, `mnist`, `cifar100`
- `--data-dir`: Directory to store/load datasets
- `--batch-size`: Training batch size

**Training Options:**
- `--method`: DP method (`standard`, `dp_sgd_custom`, `pate`, `output_perturbation`)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--pretrained`: Use pretrained ResNet weights

**Privacy Parameters:**
- `--epsilon`: Privacy budget (lower = more private)
- `--delta`: Privacy parameter (typically 1e-5)
- `--max-grad-norm`: Gradient clipping threshold for DP-SGD
- `--noise-multiplier`: Noise scaling for DP-SGD

**PATE Specific:**
- `--num-teachers`: Number of teacher models
- `--teacher-epochs`: Epochs for teacher training
- `--student-epochs`: Epochs for student training

**Output Options:**
- `--output-dir`: Directory to save results
- `--model-name`: Custom model name
- `--no-visualizations`: Skip generating plots

#### Examples

1. **Standard Training (Baseline)**:
```bash
python scripts/train_with_dp.py \
    --method standard \
    --dataset cifar10 \
    --epochs 30 \
    --lr 0.001 \
    --output-dir ./results/baseline
```

2. **DP-SGD Training**:
```bash
python scripts/train_with_dp.py \
    --method dp_sgd_custom \
    --dataset cifar10 \
    --epochs 30 \
    --epsilon 1.0 \
    --delta 1e-5 \
    --max-grad-norm 1.0 \
    --noise-multiplier 1.1 \
    --output-dir ./results/dp_sgd
```

3. **PATE Training**:
```bash
python scripts/train_with_dp.py \
    --method pate \
    --dataset mnist \
    --num-teachers 10 \
    --teacher-epochs 15 \
    --student-epochs 20 \
    --epsilon 8.0 \
    --output-dir ./results/pate
```

### Membership Inference Attacks

The `mia_runner.py` script evaluates privacy through various attacks:

#### Command Line Arguments

**Model and Data:**
- `--model`: Path to trained model checkpoint
- `--dataset`: Dataset used for training
- `--data-dir`: Dataset directory

**Attack Options:**
- `--attack`: Attack type (`shadow`, `reference`, `loo`, `all`)
- `--attack-size`: Number of samples for evaluation

**Shadow Attack:**
- `--num-shadows`: Number of shadow models to train
- `--shadow-epochs`: Training epochs for shadow models
- `--attack-epochs`: Training epochs for attack model

**Reference Attack:**
- `--num-references`: Number of reference models
- `--reference-epochs`: Training epochs for reference models

#### Examples

1. **Shadow Model Attack**:
```bash
python scripts/mia_runner.py \
    --model results/cifar10_dp_sgd_eps1.0.pt \
    --dataset cifar10 \
    --attack shadow \
    --num-shadows 5 \
    --shadow-epochs 10 \
    --attack-size 1000
```

2. **Reference Model Attack**:
```bash
python scripts/mia_runner.py \
    --model results/mnist_pate.pt \
    --dataset mnist \
    --attack reference \
    --num-references 3 \
    --reference-epochs 10
```

3. **Comprehensive Evaluation**:
```bash
python scripts/mia_runner.py \
    --model results/cifar10_standard.pt \
    --dataset cifar10 \
    --attack all \
    --output-dir ./attack_results/comprehensive
```

## Understanding Results

### Training Outputs

Each training run produces:
- **Model checkpoint**: `.pt` file with trained weights
- **Training history**: `.json` file with loss/accuracy curves
- **Summary**: `.json` file with final metrics
- **Visualizations**: Training plots (if enabled)

### Attack Results

Attack evaluation provides:
- **AUC Score**: Area under ROC curve (0.5 = random, 1.0 = perfect attack)
- **Accuracy**: Binary classification accuracy for membership prediction
- **Privacy Risk Assessment**: HIGH (>0.8), MEDIUM (0.6-0.8), LOW (<0.6)
- **Performance Plots**: ROC and Precision-Recall curves

### Privacy-Utility Trade-offs

- **Lower ε (epsilon)**: Better privacy, potentially lower utility
- **Higher ε (epsilon)**: Worse privacy, potentially better utility
- **DP-SGD**: Good privacy-utility balance, slower training
- **PATE**: Excellent privacy for small datasets, requires multiple models
- **Output Perturbation**: Simple but limited privacy guarantees

## Architecture Details

### Differential Privacy Methods

1. **DP-SGD**: Clips gradients and adds calibrated noise during training
2. **PATE**: Trains multiple teacher models on disjoint data, uses noisy aggregation
3. **Output Perturbation**: Adds noise to model outputs during training

### Attack Implementations

1. **Shadow Models**: Train surrogate models to mimic target behavior
2. **Reference Models**: Compare target confidence with reference distributions
3. **Leave-One-Out**: Analyze confidence differences when samples are excluded

### Model Architecture

- **ResNet-18**: Default architecture for image classification
- **Pretrained Options**: Can use ImageNet pretrained weights
- **Flexible**: Easy to extend with custom architectures

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black scripts/ mia_lib/
flake8 scripts/ mia_lib/
```

### Adding New DP Methods

1. Implement method in `mia_lib/dp_training.py`
2. Add command line support in `scripts/train_with_dp.py`
3. Update documentation

### Adding New Attacks

1. Create attack module in `mia_lib/attack/`
2. Integrate with `scripts/mia_runner.py`
3. Add evaluation metrics
