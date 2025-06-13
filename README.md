# Privacy-Preserving Machine Learning

This project implements and evaluates privacy-preserving machine learning techniques, focusing on differential privacy and membership inference attacks.

## Overview

The project provides tools for:
1. Training models with differential privacy
2. Evaluating model privacy using membership inference attacks
3. Implementing various defense mechanisms
4. Analyzing privacy-utility trade-offs

## Project Structure

```
.
├── docs/                      # Documentation
│   ├── differential_privacy.md
│   └── membership_inference.md
├── src/                       # Source code
│   ├── models/               # Model implementations
│   ├── attacks/              # Attack implementations
│   ├── defenses/             # Defense mechanisms
│   └── utils/                # Utility functions
├── tests/                    # Test files
├── data/                     # Data directory
├── results/                  # Results directory
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models with Differential Privacy

```bash
python src/train.py \
    --model-type cnn \
    --dataset cifar10 \
    --dp-method dp-sgd \
    --epsilon 1.0 \
    --delta 1e-5 \
    --batch-size 128 \
    --epochs 100 \
    --output-dir results/dp_models/
```

### Running Membership Inference Attacks

```bash
python src/attacks/membership_inference_attacks.py \
    --attack-type shadow \
    --target-model results/dp_models/model.pt \
    --train-data data/train.pt \
    --test-data data/test.pt \
    --output-dir results/attacks/ \
    --batch-size 128 \
    --num-shadow-models 5 \
    --attack-epochs 10
```

### Evaluating Defense Mechanisms

```bash
python src/evaluate_defenses.py \
    --model-path results/dp_models/model.pt \
    --attack-results results/attacks/ \
    --defense-methods dp-sgd,regularization \
    --output-dir results/defenses/
```

## Key Features

### Differential Privacy Methods

1. **DP-SGD (Differential Private Stochastic Gradient Descent)**
   - Gradient clipping
   - Noise addition
   - Privacy accounting

2. **PATE (Private Aggregation of Teacher Ensembles)**
   - Teacher model training
   - Private aggregation
   - Student model distillation

3. **DP-FedAvg (Differential Private Federated Averaging)**
   - Local model training
   - Private model aggregation
   - Privacy budget management

### Membership Inference Attacks

1. **Shadow Model Attack**
   - Synthetic data generation
   - Shadow model training
   - Attack model training

2. **Threshold Attack**
   - Confidence-based inference
   - Threshold optimization
   - Performance evaluation

3. **Loss-Based Attack**
   - Loss computation
   - Threshold-based inference
   - Attack evaluation

### Defense Mechanisms

1. **Differential Privacy**
   - DP-SGD implementation
   - Privacy budget tracking
   - Noise calibration

2. **Regularization**
   - L2 regularization
   - Dropout
   - Early stopping

3. **Model Modifications**
   - Model capacity reduction
   - Ensemble methods
   - Knowledge distillation

## Documentation

Detailed documentation is available in the `docs/` directory:
- [Differential Privacy](docs/differential_privacy.md)
- [Membership Inference Attacks](docs/membership_inference.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Abadi, M., et al. (2016). Deep Learning with Differential Privacy
2. Shokri, R., et al. (2017). Membership Inference Attacks against Machine Learning Models
3. Papernot, N., et al. (2017). Semi-supervised Knowledge Transfer for Deep Learning
4. McMahan, B., et al. (2018). A General Approach to Adding Differential Privacy
5. Nasr, M., et al. (2019). Comprehensive Privacy Analysis of Deep Learning

[https://programming-dp.com/intro.html](DP book)
[Membership Inference Attacks Against
Machine Learning Models](https://arxiv.org/pdf/1610.05820)
[Youtube Video](https://www.youtube.com/watch?v=rDm1n2gceJY&t=832s&ab_channel=IEEESymposiumonSecurityandPrivacy)
[Advances in Differential Privacy and Differentially Private Machine Learning](https://arxiv.org/abs/2404.04706)
[Differential Privacy and Machine Learning:
a Survey and Review](https://arxiv.org/pdf/1412.7584)
[On the Vulnerability of Data Points under
Multiple Membership Inference Attacks and
Target Models](https://arxiv.org/pdf/2210.16258)
[Membership Inference Attacks against Machine Learning Models](https://arxiv.org/abs/1610.05820)

Code Stolen from [this implementation](https://github.com/yonsei-sslab/MIA)


[MIA-repo](https://github.com/HongshengHu/membership-inference-machine-learning-literature)

[OSLO](https://arxiv.org/pdf/2007.14321)
[Label-only](https://github.com/cchoquette/membership-inference)
*** [membership-inference-evaluation](https://github.com/inspire-group/membership-inference-evaluation)
*** [LDC_MIA](https://github.com/horanshi/LDC-MIA)
*** [Quantile-MIA](https://github.com/amazon-science/quantile-mia)
*** [Enhanced-MIA](https://arxiv.org/pdf/2111.09679)
[Subpopulation based MIA](https://arxiv.org/pdf/2203.02080)