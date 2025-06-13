#!/usr/bin/env python3
"""
Quick test script to verify the loss-based attack fix.
"""

import subprocess
import sys
from pathlib import Path

def test_loss_attack():
    """Test the loss-based attack to make sure it works."""
    print("🧪 Testing Loss-based Attack Fix")
    print("="*50)
    
    # Check if we have a trained model to test with
    model_paths = [
        "./results/output_pert/cifar10_output_perturbation_eps1.0.pt",
        "./results/baseline/cifar10_standard_eps1.0.pt",
        "./test_results/test_standard.pt",
        "./test_results/test_dp.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found. Please train a model first:")
        print("python scripts/train_with_dp.py --method standard --dataset cifar10 --epochs 2 --output-dir ./test_results --model-name test_model")
        return False
    
    print(f"✅ Found model: {model_path}")
    
    # Test loss attack
    cmd = (
        f"python scripts/simple_mia_runner.py "
        f"--model-path {model_path} "
        f"--dataset cifar10 "
        f"--attack-type loss "
        f"--output-dir ./test_results/loss_attack_test"
    )
    
    print(f"Running command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Loss-based attack completed successfully!")
        print("\nOutput:")
        print(result.stdout[-1000:])  # Last 1000 chars
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Loss-based attack failed!")
        print("Error:", e.stderr)
        return False

if __name__ == "__main__":
    success = test_loss_attack()
    if success:
        print("\n🎉 Loss-based attack is now working correctly!")
        print("The 'average=binary' error has been fixed.")
    else:
        print("\n❌ There are still issues with the loss-based attack.")
    
    sys.exit(0 if success else 1) 