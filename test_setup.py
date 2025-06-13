#!/usr/bin/env python3
"""
Simple test script to verify the codebase works properly.

This script runs a quick test of the main functionality:
1. Train a simple standard model
2. Train a simple DP model
3. Run basic attacks

Usage:
    python test_setup.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False

def main():
    """Run basic tests of the system."""
    print("🧪 Testing Cleaned Up Differential Privacy and MIA System")
    print("This will run quick tests with minimal epochs to verify functionality.")
    
    # Create test output directory
    test_dir = Path("./test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Test 1: Train standard model
    cmd1 = (
        "python scripts/train_with_dp.py "
        "--method standard "
        "--dataset cifar10 "
        "--epochs 2 "
        "--batch-size 64 "
        "--output-dir ./test_results "
        "--model-name test_standard"
    )
    
    success1 = run_command(cmd1, "Standard Model Training")
    
    if not success1:
        print("\n❌ Standard training failed. Check your setup.")
        return False
    
    # Test 2: Train DP model
    cmd2 = (
        "python scripts/train_with_dp.py "
        "--method dp_sgd_custom "
        "--dataset cifar10 "
        "--epochs 2 "
        "--epsilon 1.0 "
        "--delta 1e-5 "
        "--batch-size 64 "
        "--output-dir ./test_results "
        "--model-name test_dp"
    )
    
    success2 = run_command(cmd2, "DP-SGD Model Training")
    
    if not success2:
        print("\n❌ DP training failed. Check your DP implementation.")
        return False
    
    # Test 3: Run threshold attack
    cmd3 = (
        "python scripts/simple_mia_runner.py "
        "--model-path ./test_results/test_standard.pt "
        "--dataset cifar10 "
        "--attack-type threshold "
        "--output-dir ./test_results/attacks"
    )
    
    success3 = run_command(cmd3, "Threshold Attack")
    
    if not success3:
        print("\n❌ Threshold attack failed. Check your attack implementation.")
        return False
    
    # Test 4: Run loss attack
    cmd4 = (
        "python scripts/simple_mia_runner.py "
        "--model-path ./test_results/test_dp.pt "
        "--dataset cifar10 "
        "--attack-type loss "
        "--output-dir ./test_results/attacks_dp"
    )
    
    success4 = run_command(cmd4, "Loss-based Attack")
    
    if not success4:
        print("\n❌ Loss attack failed. Check your attack implementation.")
        return False
    
    # All tests passed
    print(f"\n{'='*60}")
    print("🎉 ALL TESTS PASSED!")
    print("✅ Standard training works")
    print("✅ DP-SGD training works") 
    print("✅ Threshold attack works")
    print("✅ Loss-based attack works")
    print(f"{'='*60}")
    print("\nYour system is ready to use!")
    print("Check the test_results/ directory for output files.")
    print("\nNext steps:")
    print("1. Run longer training with more epochs")
    print("2. Try different privacy budgets")
    print("3. Test shadow attacks (they take longer)")
    print("4. Compare attack success rates between standard and DP models")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 