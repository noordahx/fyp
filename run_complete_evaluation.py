#!/usr/bin/env python3
"""
Complete DP Training and MIA Evaluation Pipeline

This script will:
1. Train target models with different privacy methods (standard, DP-SGD, output perturbation)
2. Run all MIA attacks on each trained model
3. Save comprehensive results to colab_results/ folder
4. Generate summary reports

Usage:
    python run_complete_evaluation.py
    python run_complete_evaluation.py --config configs/full_config.yaml
    python run_complete_evaluation.py --dataset cifar10 --epochs 30
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath('.'))

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("colab_results") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def run_command(command, description, logger):
    """Run a command and log the results with real-time output."""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    start_time = time.time()
    
    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=dict(os.environ, PYTHONUNBUFFERED='1')  # Ensure unbuffered Python output
        )
        
        output_lines = []
        print(f"🔄 {description} - Live Output:")
        print("-" * 60)
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')  # Print to console immediately
            sys.stdout.flush()   # Force immediate display
            output_lines.append(line)
            
        process.wait()
        print("-" * 60)
        
        elapsed = time.time() - start_time
        full_output = ''.join(output_lines)
        
        if process.returncode == 0:
            logger.info(f"✅ Completed: {description} in {elapsed:.1f}s")
            return True, full_output, ''
        else:
            logger.error(f"❌ Failed: {description} after {elapsed:.1f}s")
            logger.error(f"Return code: {process.returncode}")
            return False, full_output, f"Process exited with code {process.returncode}"
            
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Failed: {description} after {elapsed:.1f}s")
        logger.error(f"Error: {str(e)}")
        return False, '', str(e)

def train_target_models(config_path, output_base_dir, logger):
    """Train target models with different privacy methods."""
    
    # Define privacy methods and their configurations
    privacy_methods = [
        {
            'name': 'standard',
            'method': 'standard',
            'epsilon': 'inf',  # No privacy
            'description': 'Standard training (no privacy)'
        },
        {
            'name': 'dp_sgd_eps3',
            'method': 'dp_sgd_custom',
            'epsilon': 3.0,
            'description': 'DP-SGD with ε=3.0 (low privacy)'
        },
        {
            'name': 'dp_sgd_eps1',
            'method': 'dp_sgd_custom', 
            'epsilon': 1.0,
            'description': 'DP-SGD with ε=1.0 (medium privacy)'
        },
        {
            'name': 'dp_sgd_eps0_5',
            'method': 'dp_sgd_custom',
            'epsilon': 0.5,
            'description': 'DP-SGD with ε=0.5 (high privacy)'
        },
        {
            'name': 'output_pert_eps3',
            'method': 'output_perturbation',
            'epsilon': 3.0,
            'description': 'Output Perturbation with ε=3.0'
        },
        {
            'name': 'output_pert_eps1',
            'method': 'output_perturbation',
            'epsilon': 1.0,
            'description': 'Output Perturbation with ε=1.0'
        }
    ]
    
    trained_models = []
    models_dir = Path(output_base_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 Starting training of {len(privacy_methods)} target models...")
    
    for i, method_config in enumerate(privacy_methods, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Model {i}/{len(privacy_methods)}: {method_config['description']}")
        logger.info(f"{'='*60}")
        
        # Prepare output directory for this method
        method_output_dir = models_dir / method_config['name']
        
        # Build training command
        cmd_parts = [
            "python scripts/train.py",
            f"--config {config_path}",
            f"--method {method_config['method']}",
            f"--output_dir {method_output_dir}"
        ]
        
        # Add epsilon parameter for privacy methods
        if method_config['method'] != 'standard':
            if isinstance(method_config['epsilon'], (int, float)):
                cmd_parts.append(f"--epsilon {method_config['epsilon']}")
        
        command = " ".join(cmd_parts)
        
        success, stdout, stderr = run_command(
            command, 
            f"Training {method_config['name']}", 
            logger
        )
        
        if success:
            # Find the trained model file
            model_files = list(method_output_dir.glob("*.pt"))
            if model_files:
                model_path = model_files[0]
                trained_models.append({
                    'name': method_config['name'],
                    'method': method_config['method'],
                    'epsilon': method_config['epsilon'],
                    'description': method_config['description'],
                    'model_path': str(model_path),
                    'training_success': True
                })
                logger.info(f"✅ Model saved: {model_path}")
            else:
                logger.error(f"❌ No model file found in {method_output_dir}")
                trained_models.append({
                    'name': method_config['name'],
                    'training_success': False,
                    'error': 'Model file not found'
                })
        else:
            trained_models.append({
                'name': method_config['name'],
                'training_success': False,
                'error': stderr
            })
    
    # Save training summary
    summary_file = Path(output_base_dir) / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(trained_models, f, indent=2)
    
    successful_models = [m for m in trained_models if m.get('training_success', False)]
    logger.info(f"\n🎯 Training Summary: {len(successful_models)}/{len(privacy_methods)} models trained successfully")
    
    return successful_models

def run_mia_attacks(trained_models, config_path, output_base_dir, logger):
    """Run all MIA attacks on each trained model."""
    
    attacks_dir = Path(output_base_dir) / "attacks"
    attacks_dir.mkdir(parents=True, exist_ok=True)
    
    attack_results = []
    total_attacks = len(trained_models) * 4  # 4 attack types per model
    current_attack = 0
    
    logger.info(f"\n🎯 Starting MIA evaluation on {len(trained_models)} models...")
    logger.info(f"Total attacks to run: {total_attacks}")
    
    for model_info in trained_models:
        if not model_info.get('training_success', False):
            logger.warning(f"⚠️ Skipping attacks for {model_info['name']} - training failed")
            continue
            
        model_name = model_info['name']
        model_path = model_info['model_path']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running attacks on: {model_name}")
        logger.info(f"Model: {model_path}")
        logger.info(f"{'='*60}")
        
        # Create output directory for this model's attacks
        model_attack_dir = attacks_dir / model_name
        model_attack_dir.mkdir(parents=True, exist_ok=True)
        
        # Run all attacks
        attack_types = ['threshold', 'loss', 'population', 'shadow']
        
        for attack_type in attack_types:
            current_attack += 1
            logger.info(f"\n[{current_attack}/{total_attacks}] Running {attack_type.upper()} attack on {model_name}...")
            
            # Build attack command
            command = (
                f"python scripts/attack.py "
                f"--config {config_path} "
                f"--model {model_path} "
                f"--attack {attack_type} "
                f"--output-dir {model_attack_dir}"
            )
            
            success, stdout, stderr = run_command(
                command,
                f"{attack_type} attack on {model_name}",
                logger
            )
            
            # Record results
            result_entry = {
                'model_name': model_name,
                'model_method': model_info['method'],
                'model_epsilon': model_info['epsilon'],
                'attack_type': attack_type,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                # Try to parse attack results from output or result files
                result_file = model_attack_dir / f"{attack_type}_results.json"
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            attack_data = json.load(f)
                        
                        result_entry.update({
                            'auc': attack_data.get('auc', 0.0),
                            'accuracy': attack_data.get('accuracy', 0.0),
                            'attack_advantage': attack_data.get('attack_advantage', 0.0),
                            'precision': attack_data.get('precision', 0.0),
                            'recall': attack_data.get('recall', 0.0),
                            'f1': attack_data.get('f1', 0.0)
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse results for {attack_type}: {e}")
                        result_entry['parse_error'] = str(e)
            else:
                result_entry['error'] = stderr
            
            attack_results.append(result_entry)
    
    # Save attack results summary
    results_summary_file = Path(output_base_dir) / "attack_results_summary.json"
    with open(results_summary_file, 'w') as f:
        json.dump(attack_results, f, indent=2)
    
    # Create CSV summary for easy analysis
    if attack_results:
        df = pd.DataFrame(attack_results)
        csv_file = Path(output_base_dir) / "attack_results_summary.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"📊 Results saved to: {csv_file}")
    
    successful_attacks = len([r for r in attack_results if r.get('success', False)])
    logger.info(f"\n🎯 Attack Summary: {successful_attacks}/{len(attack_results)} attacks completed successfully")
    
    return attack_results

def generate_final_report(trained_models, attack_results, output_base_dir, logger):
    """Generate a comprehensive final report."""
    
    report_file = Path(output_base_dir) / "final_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DIFFERENTIAL PRIVACY AND MIA EVALUATION REPORT\n") 
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Training Summary
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 40 + "\n")
        successful_models = [m for m in trained_models if m.get('training_success', False)]
        f.write(f"Models trained: {len(successful_models)}/{len(trained_models)}\n\n")
        
        for model in successful_models:
            f.write(f"✅ {model['name']}: {model['description']}\n")
        
        failed_models = [m for m in trained_models if not m.get('training_success', False)]
        if failed_models:
            f.write(f"\nFailed models: {len(failed_models)}\n")
            for model in failed_models:
                f.write(f"❌ {model['name']}\n")
        
        # Attack Summary
        f.write("\n\nATTACK RESULTS SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        if attack_results:
            # Group by model
            models_dict = {}
            for result in attack_results:
                model_name = result['model_name']
                if model_name not in models_dict:
                    models_dict[model_name] = []
                models_dict[model_name].append(result)
            
            for model_name, model_attacks in models_dict.items():
                f.write(f"\nModel: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                for attack in model_attacks:
                    if attack.get('success', False):
                        auc = attack.get('auc', 0.0)
                        acc = attack.get('accuracy', 0.0)
                        adv = attack.get('attack_advantage', 0.0)
                        f.write(f"{attack['attack_type'].upper():<12}: AUC={auc:.4f}, Acc={acc:.4f}, Adv={adv:.4f}\n")
                    else:
                        f.write(f"{attack['attack_type'].upper():<12}: FAILED\n")
        
        # Privacy Analysis
        f.write("\n\nPRIVACY ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Higher AUC/Accuracy/Advantage = More vulnerable to attacks = Less private\n")
        f.write("Lower values = More protected by differential privacy\n\n")
        
        f.write("Expected trends:\n")
        f.write("- Standard model: Highest vulnerability (AUC > 0.8)\n")
        f.write("- DP-SGD ε=3.0: Moderate vulnerability (AUC ~0.6-0.7)\n")
        f.write("- DP-SGD ε=1.0: Lower vulnerability (AUC ~0.5-0.6)\n")
        f.write("- DP-SGD ε=0.5: Lowest vulnerability (AUC ~0.5)\n")
        f.write("- Output Perturbation: Variable results depending on implementation\n")
    
    logger.info(f"📋 Final report saved: {report_file}")

def main():
    """Main function to run the complete evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete DP Training and MIA Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete_evaluation.py
  python run_complete_evaluation.py --config configs/full_config.yaml  
  python run_complete_evaluation.py --output-dir ./my_results
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/full_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='colab_results',
                        help='Output directory for all results')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training phase (use existing models)')
    parser.add_argument('--skip-attacks', action='store_true', 
                        help='Skip attack phase (only train models)')
    
    args = parser.parse_args()
    
    # Setup
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging()
    
    # Start evaluation
    start_time = time.time()
    logger.info("🚀 Starting Complete DP Training and MIA Evaluation Pipeline")
    logger.info(f"📁 Output directory: {output_base_dir.absolute()}")
    logger.info(f"⚙️ Config file: {args.config}")
    
    try:
        # Phase 1: Train target models
        if not args.skip_training:
            logger.info("\n" + "="*80)
            logger.info("PHASE 1: TRAINING TARGET MODELS")
            logger.info("="*80)
            
            trained_models = train_target_models(args.config, output_base_dir, logger)
        else:
            logger.info("⏭️ Skipping training phase")
            # Load existing training summary
            summary_file = output_base_dir / "training_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    trained_models = json.load(f)
            else:
                logger.error("❌ No training summary found. Cannot skip training.")
                return
        
        # Phase 2: Run MIA attacks
        if not args.skip_attacks:
            logger.info("\n" + "="*80)
            logger.info("PHASE 2: RUNNING MIA ATTACKS")
            logger.info("="*80)
            
            attack_results = run_mia_attacks(trained_models, args.config, output_base_dir, logger)
        else:
            logger.info("⏭️ Skipping attack phase")
            attack_results = []
        
        # Phase 3: Generate final report
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: GENERATING FINAL REPORT")
        logger.info("="*80)
        
        generate_final_report(trained_models, attack_results, output_base_dir, logger)
        
        # Complete
        total_time = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("🎉 EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        logger.info(f"📁 All results saved in: {output_base_dir.absolute()}")
        logger.info("="*80)
        
        print(f"\n🎯 Quick Access:")
        print(f"📊 Results Summary: {output_base_dir}/attack_results_summary.csv")
        print(f"📋 Final Report: {output_base_dir}/final_report.txt")
        print(f"📁 All Files: {output_base_dir.absolute()}")
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()