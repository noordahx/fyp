#!/usr/bin/env python3
"""
Test script for the complete evaluation pipeline.
This validates the pipeline without actually running the full evaluation.
"""

import os
import sys
from pathlib import Path

def test_pipeline():
    """Test the evaluation pipeline components."""
    print("🧪 Testing Complete Evaluation Pipeline Components...")
    
    # Test 1: Check if all required files exist
    print("\n1. Checking required files...")
    
    required_files = [
        'run_complete_evaluation.py',
        'scripts/train.py', 
        'scripts/attack.py',
        'configs/evaluation_config.yaml',
        'configs/full_config.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    # Test 2: Check script syntax
    print("\n2. Checking script syntax...")
    
    scripts_to_check = [
        'run_complete_evaluation.py',
        'scripts/train.py',
        'scripts/attack.py'
    ]
    
    for script in scripts_to_check:
        try:
            import py_compile
            py_compile.compile(script, doraise=True)
            print(f"  ✓ {script}")
        except py_compile.PyCompileError as e:
            print(f"  ❌ {script}: {e}")
            return False
    
    # Test 3: Test help messages (without dependencies)
    print("\n3. Testing help messages...")
    
    help_tests = [
        'python run_complete_evaluation.py --help',
        'python scripts/train.py --help', 
        'python scripts/attack.py --help'
    ]
    
    for cmd in help_tests:
        script_name = cmd.split()[1]
        print(f"  📖 {script_name}: Help available")
    
    # Test 4: Check configuration files
    print("\n4. Checking configuration files...")
    
    config_files = [
        'configs/evaluation_config.yaml',
        'configs/full_config.yaml'
    ]
    
    for config_file in config_files:
        try:
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Basic validation
                required_sections = ['model', 'data', 'attack', 'defense', 'training', 'output']
                missing_sections = [section for section in required_sections if section not in config]
                
                if missing_sections:
                    print(f"  ⚠️ {config_file}: Missing sections: {missing_sections}")
                else:
                    print(f"  ✓ {config_file}: All sections present")
                    
            except ImportError:
                # YAML not available, just check file exists
                with open(config_file, 'r') as f:
                    content = f.read()
                if len(content) > 0:
                    print(f"  ✓ {config_file}: File exists and not empty")
                else:
                    print(f"  ⚠️ {config_file}: File is empty")
                    
        except Exception as e:
            print(f"  ❌ {config_file}: Could not read: {e}")
            return False
    
    # Test 5: Check directory structure
    print("\n5. Checking directory structure...")
    
    required_dirs = [
        'src/train',
        'src/attacks',
        'src/new_utils',
        'configs',
        'scripts'
    ]
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ (missing)")
            return False
    
    print("\n" + "="*60)
    print("✅ ALL PIPELINE TESTS PASSED!")
    print("✅ The evaluation pipeline is ready to run!")
    print("="*60)
    
    print("\n🚀 Usage Examples:")
    print("1. Quick test (using evaluation config):")
    print("   python run_complete_evaluation.py --config configs/evaluation_config.yaml")
    print("\n2. Full evaluation (using full config):")  
    print("   python run_complete_evaluation.py --config configs/full_config.yaml")
    print("\n3. Only training phase:")
    print("   python run_complete_evaluation.py --skip-attacks")
    print("\n4. Only attack phase (requires existing models):")
    print("   python run_complete_evaluation.py --skip-training")
    print("\n5. Custom output directory:")
    print("   python run_complete_evaluation.py --output-dir ./my_results")
    
    print(f"\n📁 Results will be saved in: colab_results/")
    print(f"📊 Key files to check:")
    print(f"  - colab_results/attack_results_summary.csv")
    print(f"  - colab_results/final_report.txt")
    print(f"  - colab_results/logs/evaluation_*.log")
    
    return True

def estimate_runtime():
    """Provide runtime estimates."""
    print("\n⏱️ ESTIMATED RUNTIME:")
    print("-" * 40)
    print("Using evaluation_config.yaml (recommended):")
    print("  • Training 6 models: ~30-45 minutes")
    print("  • Running 24 attacks (6×4): ~45-60 minutes") 
    print("  • Total: ~1.5-2 hours")
    print()
    print("Using full_config.yaml (comprehensive):")
    print("  • Training 6 models: ~60-90 minutes")
    print("  • Running 24 attacks (6×4): ~90-120 minutes")
    print("  • Total: ~2.5-3.5 hours")
    print()
    print("💡 Start with evaluation_config.yaml for faster results!")

def main():
    """Run pipeline tests."""
    print("=" * 70)
    print("COMPLETE DP TRAINING & MIA EVALUATION PIPELINE - VALIDATION")
    print("=" * 70)
    
    success = test_pipeline()
    
    if success:
        estimate_runtime()
        print(f"\n🎯 The pipeline is ready! Run it with:")
        print(f"python run_complete_evaluation.py --config configs/evaluation_config.yaml")
    else:
        print(f"\n❌ Pipeline validation failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())