#!/usr/bin/env python3
"""
Run training with predefined configurations for high MIA AUC.

This script loads configurations from configs/high_auc_configs.json and
runs the training script with the specified parameters.

Usage:
    python scripts/run_config.py --config cifar10_high_auc
    python scripts/run_config.py --config cifar10_extreme_auc --output-dir ./results/extreme
    python scripts/run_config.py --list-configs  # List available configurations
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

def load_configs():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / 'configs' / 'high_auc_configs.json'
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        return json.load(f)

def list_configs(configs):
    """List available configurations."""
    print("Available configurations:")
    print("=" * 50)
    
    for name, config in configs['configs'].items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Dataset: {config['dataset']}")
        print(f"  Model size: {config['model_size']}")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Expected AUC: {config['expected_auc']}")
        print(f"  GPU memory: {config['gpu_memory_required']}")
        print(f"  Training time: {config['training_time_estimate']}")

def run_config(config_name, configs, output_dir=None, dry_run=False):
    """Run training with specified configuration."""
    
    if config_name not in configs['configs']:
        print(f"Error: Configuration '{config_name}' not found.")
        print("Available configurations:", list(configs['configs'].keys()))
        sys.exit(1)
    
    config = configs['configs'][config_name]
    
    # Build command
    script_path = Path(__file__).parent / 'train_high_auc_model.py'
    cmd = ['python', str(script_path)]
    
    # Add configuration parameters
    cmd.extend(['--dataset', config['dataset']])
    cmd.extend(['--model-size', config['model_size']])
    cmd.extend(['--epochs', str(config['epochs'])])
    cmd.extend(['--batch-size', str(config['batch_size'])])
    cmd.extend(['--lr', str(config['lr'])])
    cmd.extend(['--weight-decay', str(config['weight_decay'])])
    cmd.extend(['--eval-size', str(config['eval_size'])])
    
    # Add optional parameters
    if config.get('train_size'):
        cmd.extend(['--train-size', str(config['train_size'])])
    
    if config.get('test_size'):
        cmd.extend(['--test-size', str(config['test_size'])])
    
    # Add output directory
    if output_dir:
        cmd.extend(['--output-dir', output_dir])
    else:
        cmd.extend(['--output-dir', f'./results/{config_name}'])
    
    # Add model name
    cmd.extend(['--model-name', config_name])
    
    print(f"Running configuration: {config_name}")
    print(f"Description: {config['description']}")
    print(f"Expected AUC: {config['expected_auc']}")
    print(f"GPU memory required: {config['gpu_memory_required']}")
    print(f"Estimated training time: {config['training_time_estimate']}")
    print()
    
    if dry_run:
        print("Dry run - command that would be executed:")
        print(' '.join(cmd))
        return
    
    print("Command:", ' '.join(cmd))
    print("Starting training...")
    print("=" * 60)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("=" * 60)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run training with predefined configurations')
    
    parser.add_argument('--config', type=str,
                       help='Configuration name to run')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configurations')
    parser.add_argument('--output-dir', type=str,
                       help='Override output directory')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show command without executing')
    
    args = parser.parse_args()
    
    # Load configurations
    configs = load_configs()
    
    if args.list_configs:
        list_configs(configs)
        return
    
    if not args.config:
        print("Error: Please specify a configuration with --config or use --list-configs")
        print("Available configurations:", list(configs['configs'].keys()))
        sys.exit(1)
    
    run_config(args.config, configs, args.output_dir, args.dry_run)

if __name__ == '__main__':
    main() 