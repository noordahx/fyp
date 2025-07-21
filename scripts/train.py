#!/usr/bin/env python3
"""
Differential Privacy Training Script

Usage:
    python scripts/train.py --config configs/simple_config.yaml
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
import torch
from pathlib import Path

from src.new_utils.config import load_config_from_yaml
from src.train.dp_training import run_training

# Setup logging to be visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='Train model with (or without) Differential Privacy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --config configs/simple_config.yaml --method standard
  python scripts/train.py --config configs/simple_config.yaml --method dp_sgd_custom --epsilon 1.0
  python scripts/train.py --config configs/simple_config.yaml --method output_perturbation --epsilon 2.0 --output_dir ./my_models
        """
    )
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--method', type=str, default=None, 
                        help='Override privacy method: standard, dp_sgd_custom, output_perturbation')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--epsilon', type=float, default=None, 
                        help='Privacy budget (epsilon) - overrides config value')
    parser.add_argument('--delta', type=float, default=None,
                        help='Privacy parameter (delta) - overrides config value')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Gradient clipping norm - overrides config value')
    parser.add_argument('--noise_multiplier', type=float, default=None,
                        help='Noise multiplier - overrides config value')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs - overrides config value')
    
    args = parser.parse_args()

    # Load config from YAML
    config = load_config_from_yaml(args.config)

    # Override configuration with command line arguments
    if args.method:
        valid_methods = ['standard', 'dp_sgd_custom', 'output_perturbation']
        if args.method not in valid_methods:
            raise ValueError(f"Unknown method: {args.method}. Available methods: {valid_methods}")
        config.defense.method = args.method
    
    if args.output_dir:
        config.output.save_dir = args.output_dir
        
    if args.epsilon is not None:
        config.defense.epsilon = args.epsilon
        
    if args.delta is not None:
        config.defense.delta = args.delta
        
    if args.max_grad_norm is not None:
        config.defense.max_grad_norm = args.max_grad_norm
        
    if args.noise_multiplier is not None:
        config.defense.noise_multiplier = args.noise_multiplier
        
    if args.epochs is not None:
        config.training.epochs = args.epochs
    logger.info(f"Using output directory: {config.output.save_dir}")

    output_dir = Path(config.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # GPU memory and performance info
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.3f} GB")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        logger.info("GPU optimizations enabled: cudnn.benchmark=True")

    # Run the main training routine
    run_training(config=config, device=device)

if __name__ == '__main__':
    main()