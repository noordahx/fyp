#!/usr/bin/env python3
"""
Generate visual reports from existing training results.

This script can generate comprehensive visual reports from saved training results,
including training analysis, MIA attack results, and executive dashboards.

Usage:
    python scripts/generate_visual_report.py --results-file results/high_auc/model_results.json
    python scripts/generate_visual_report.py --results-dir results/high_auc --pattern "*_results.json"
"""

import argparse
import json
import sys
from pathlib import Path
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mia_lib.visualization import MIAVisualizer

def load_results(results_path):
    """Load results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading results from {results_path}: {e}")
        return None

def generate_reports_from_results(results, output_dir, model_name=None):
    """Generate visual reports from loaded results."""
    
    if not results:
        print("No results provided")
        return False
    
    # Extract required data
    training_history = results.get('training_history', {})
    mia_results = results.get('mia_results', {})
    
    if not training_history:
        print("No training history found in results")
        return False
    
    if not mia_results:
        print("No MIA results found in results")
        return False
    
    # Prepare model info
    model_info = {
        'model_name': model_name or results.get('model_name', 'Unknown'),
        'dataset': results.get('dataset', 'N/A'),
        'model_size': results.get('model_size', 'N/A'),
        'total_params': results.get('total_params', 0),
        'train_size': results.get('train_size', 0),
        'test_size': results.get('test_size', 0),
        'epochs': results.get('epochs', 0),
        'batch_size': results.get('batch_size', 0),
        'lr': results.get('lr', 0),
        'weight_decay': results.get('weight_decay', 0),
        'device': results.get('device', 'N/A'),
        'training_time': 'N/A'  # Not available from saved results
    }
    
    try:
        # Create visualizer
        viz_dir = Path(output_dir) / "visualizations"
        visualizer = MIAVisualizer(viz_dir)
        
        report_name = model_info['model_name']
        
        print(f"Generating reports for: {report_name}")
        print(f"Output directory: {viz_dir}")
        
        # Generate individual reports
        print("Generating training analysis report...")
        training_report = visualizer.create_training_report(
            training_history, model_info, f"{report_name}_training"
        )
        print(f"✓ Training report: {training_report}")
        
        print("Generating MIA attack analysis report...")
        mia_report = visualizer.create_mia_report(
            mia_results, model_info, f"{report_name}_mia"
        )
        print(f"✓ MIA report: {mia_report}")
        
        print("Generating executive summary dashboard...")
        dashboard = visualizer.create_summary_dashboard(
            training_history, mia_results, model_info, f"{report_name}_dashboard"
        )
        print(f"✓ Dashboard: {dashboard}")
        
        print("Generating comprehensive HTML report...")
        html_report = visualizer.generate_html_report(
            training_history, mia_results, model_info, save_name=report_name
        )
        print(f"✓ HTML report: {html_report}")
        
        print(f"\n🎉 All visual reports generated successfully!")
        print(f"📁 Reports location: {viz_dir}")
        print(f"🌐 Open HTML report: {html_report}")
        
        return True
        
    except Exception as e:
        print(f"Error generating visual reports: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate visual reports from training results')
    
    # Input arguments
    parser.add_argument('--results-file', type=str,
                       help='Path to results JSON file')
    parser.add_argument('--results-dir', type=str,
                       help='Directory containing results files')
    parser.add_argument('--pattern', type=str, default='*_results.json',
                       help='Pattern to match results files (used with --results-dir)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as results file/dir)')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Override model name for reports')
    
    # Processing arguments
    parser.add_argument('--batch', action='store_true',
                       help='Process all matching files in batch mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.results_file and not args.results_dir:
        print("Error: Must specify either --results-file or --results-dir")
        sys.exit(1)
    
    if args.results_file and args.results_dir:
        print("Error: Cannot specify both --results-file and --results-dir")
        sys.exit(1)
    
    # Process single file
    if args.results_file:
        results_path = Path(args.results_file)
        if not results_path.exists():
            print(f"Error: Results file not found: {results_path}")
            sys.exit(1)
        
        output_dir = args.output_dir or results_path.parent
        
        print(f"Loading results from: {results_path}")
        results = load_results(results_path)
        
        if results:
            success = generate_reports_from_results(results, output_dir, args.model_name)
            if not success:
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Process directory
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            sys.exit(1)
        
        # Find matching files
        pattern_path = results_dir / args.pattern
        results_files = glob.glob(str(pattern_path))
        
        if not results_files:
            print(f"Error: No results files found matching pattern: {pattern_path}")
            sys.exit(1)
        
        print(f"Found {len(results_files)} results files:")
        for f in results_files:
            print(f"  - {f}")
        
        output_dir = args.output_dir or results_dir
        
        # Process each file
        success_count = 0
        for results_file in results_files:
            print(f"\n{'='*60}")
            print(f"Processing: {results_file}")
            print(f"{'='*60}")
            
            results = load_results(results_file)
            if results:
                # Use file-specific model name if not overridden
                model_name = args.model_name or results.get('model_name', Path(results_file).stem)
                success = generate_reports_from_results(results, output_dir, model_name)
                if success:
                    success_count += 1
            
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed: {success_count}/{len(results_files)} files")
        
        if success_count == 0:
            sys.exit(1)

if __name__ == '__main__':
    main() 