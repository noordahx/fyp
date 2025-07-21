"""
Membership Inference Attack Runner

Usage:
    python attack.py --config config/my_config.yaml --attack shadow
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
import json
from pathlib import Path

import torch

from src.attacks.base import get_attack_by_name
from src.data import get_dataset
from src.new_utils.models import create_model
from src.new_utils.config import Config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='Run Membership Inference Attack',
        epilog='Examples:\n'
               '  python scripts/attack.py --config configs/simple_config.yaml --model results/model.pt --attack shadow\n'
               '  python scripts/attack.py --config configs/simple_config.yaml --model results/model.pt --attack all --output-dir ./attack_results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    parser.add_argument('--attack', type=str, default=None,
                        help='Override attack type from config. Options: shadow, loss, threshold, population, all')
    parser.add_argument("--model", type=str, required=True, help="Path to target model checkpoint (.pt file)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory (uses config default if not specified)")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    
    config = Config.from_yaml(args.config)
    
    if args.attack:
        valid_attacks = ['shadow', 'loss', 'threshold', 'population', 'all']
        if args.attack.lower() not in valid_attacks:
            raise ValueError(f"Invalid attack type '{args.attack}'. Valid options: {valid_attacks}")
        config.attack.attack_type = args.attack.lower()

    if args.output_dir:
        config.output.save_dir = args.output_dir
    
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Target model: {args.model}")
    logger.info(f"Attack type: {config.attack.attack_type}")
    logger.info(f"Output directory: {config.output.save_dir}")
    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_dataset, test_dataset = get_dataset(config.data.dataset, './data')
    logging.info(f"Loaded datasets: {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    
    model = create_model(config.model.num_classes, 
                         config.model.pretrained,
                         input_channels=1 if config.data.dataset == 'mnist' else 3)
    
    checkpoint = torch.load(args.model, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Prepare output directory
    output_dir = Path(config.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle "all" attack type
    if config.attack.attack_type.lower() == "all":
        logger.info("Running ALL MIA attacks...")
        from src.attacks.base import run_all_attacks
        all_results = run_all_attacks(config, model, train_dataset, test_dataset, device)
        
        # Save combined results
        combined_path = output_dir / "all_attacks_results.json"
        logger.info(f"Saving combined results to {combined_path}")
        
        # Prepare for JSON serialization
        json_results = {}
        for attack_name, results in all_results.items():
            json_results[attack_name] = {}
            for key, value in results.items():
                if hasattr(value, 'tolist'):  # numpy array
                    json_results[attack_name][key] = value.tolist()
                elif hasattr(value, 'item'):  # numpy scalar
                    json_results[attack_name][key] = value.item()
                else:
                    json_results[attack_name][key] = value
        
        # Add metadata
        json_results['metadata'] = {
            'model_path': args.model,
            'dataset': config.data.dataset,
            'attack_types': list(all_results.keys()),
            'config_path': args.config
        }
        
        # Save to JSON file
        with open(combined_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Print summary for all attacks
        print("\n" + "="*60)
        print("ALL MIA ATTACKS SUMMARY")
        print("="*60)
        for attack_name, results in all_results.items():
            if 'error' in results:
                print(f"{attack_name.upper():<12}: FAILED - {results['error']}")
            else:
                auc = results.get('auc', 0.0)
                acc = results.get('accuracy', results.get('accuracy_optimal', 0.0))
                adv = results.get('attack_advantage', 0.0)
                print(f"{attack_name.upper():<12}: AUC={auc:.4f}, Acc={acc:.4f}, Adv={adv:.4f}")
        print(f"Combined results saved to: {combined_path}")
        print("="*60)
        
        return all_results
    
    else:
        # Run single attack
        attack = get_attack_by_name(config.attack.attack_type, config)
        attack_name = getattr(attack, 'name', config.attack.attack_type)
        
        logger.info(f"Running {attack_name.upper()} MIA...")
        results = attack.run(model, train_dataset, test_dataset, device)
        
        # Save results
        output_path = output_dir / f"{attack_name}_results.json"
        logger.info(f"Saving results to {output_path}")
        
        # Prepare results for JSON serialization (convert numpy types)
        json_results = {}
        for key, value in results.items():
            if hasattr(value, 'tolist'):  # numpy array
                json_results[key] = value.tolist()
            elif hasattr(value, 'item'):  # numpy scalar
                json_results[key] = value.item()
            else:
                json_results[key] = value
        
        # Add metadata
        json_results['model_path'] = args.model
        json_results['dataset'] = config.data.dataset
        json_results['attack_type'] = attack_name
        json_results['config_path'] = args.config
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print(f"{attack_name.upper()} MIA RESULTS")
        print("="*50)
        print(f"AUC: {results['auc']:.4f}")
        print(f"Accuracy: {results.get('accuracy', results.get('accuracy_optimal', 0.0)):.4f}")
        print(f"Attack Advantage: {results['attack_advantage']:.4f}")
        if 'precision' in results:
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1-Score: {results['f1']:.4f}")
        print(f"Results saved to: {output_path}")
        print("="*50)
        
        return results

if __name__ == "__main__":
    try:
        main()
        print("\nAttack completed successfully!")
    except Exception as e:
        print(f"\nAttack failed: {str(e)}")
        logger.error(f"Attack failed with error: {str(e)}")
        sys.exit(1)