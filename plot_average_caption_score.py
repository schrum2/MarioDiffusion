#!/usr/bin/env python3
"""
Script to plot averaged caption scores across multiple runs.
Reads JSONL files from subdirectories and creates plots with optional error regions.
"""

import argparse
import json
import glob
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def find_jsonl_file(directory):
    """Find the caption_score_log*.jsonl file in the given directory."""
    pattern = os.path.join(directory, "caption_score_log_*.jsonl")
    files = glob.glob(pattern)
    if not files:
        return None
    if len(files) > 1:
        print(f"Warning: Multiple JSONL files found in {directory}, using {files[0]}")
    return files[0]

def read_jsonl_data(filepath):
    """Read JSONL file and return a pandas DataFrame."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def collect_run_data(prefix, run_ids):
    """Collect data from all specified runs."""
    all_data = {}
    
    for run_id in run_ids:
        dir_name = f"{prefix}{run_id}"
        if not os.path.exists(dir_name):
            print(f"Warning: Directory {dir_name} not found, skipping...")
            continue
            
        jsonl_file = find_jsonl_file(dir_name)
        if not jsonl_file:
            print(f"Warning: No JSONL file found in {dir_name}, skipping...")
            continue
            
        df = read_jsonl_data(jsonl_file)
        if df is None or df.empty:
            print(f"Warning: No data found in {jsonl_file}, skipping...")
            continue
            
        print(f"Loaded {len(df)} records from {jsonl_file}")
        all_data[run_id] = df
    
    if not all_data:
        print("Error: No valid data found in any runs!")
        sys.exit(1)
        
    return all_data

def aggregate_data(all_data):
    """Aggregate data across runs by epoch."""
    # Find all unique epochs across all runs
    all_epochs = set()
    for df in all_data.values():
        all_epochs.update(df['epoch'].unique())
    all_epochs = sorted(all_epochs)
    
    # Create aggregated data structure
    aggregated = []
    
    for epoch in all_epochs:
        scores = []
        for run_id, df in all_data.items():
            epoch_data = df[df['epoch'] == epoch]
            if not epoch_data.empty:
                # If multiple entries for same epoch, take the last one
                score = epoch_data.iloc[-1]['caption_score']
                scores.append(score)
        
        if scores:  # Only include epochs that have data from at least one run
            aggregated.append({
                'epoch': epoch,
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores, ddof=1) if len(scores) > 1 else 0,
                'count': len(scores)
            })
    
    return aggregated

def calculate_confidence_interval(scores, confidence=0.95):
    """Calculate confidence interval for a list of scores."""
    if len(scores) <= 1:
        return 0, 0
    
    mean = np.mean(scores)
    sem = stats.sem(scores)  # standard error of the mean
    ci = stats.t.interval(confidence, len(scores)-1, loc=mean, scale=sem)
    return ci[0] - mean, ci[1] - mean  # return as offsets from mean

def create_plot(aggregated_data, error_type=None, confidence=0.95):
    """Create the matplotlib plot."""
    epochs = [d['epoch'] for d in aggregated_data]
    means = [d['mean'] for d in aggregated_data]
    
    plt.figure(figsize=(10, 6))
    
    # Main line plot
    plt.plot(epochs, means, 'b-', linewidth=2, label='Mean Caption Score')
    
    # Add error regions if requested
    if error_type == 'std':
        stds = [d['std'] for d in aggregated_data]
        plt.fill_between(epochs, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, color='blue', label='± 1 Standard Deviation')
    
    elif error_type == 'ci':
        ci_lower = []
        ci_upper = []
        for d in aggregated_data:
            ci_low, ci_high = calculate_confidence_interval(d['scores'], confidence)
            ci_lower.append(d['mean'] + ci_low)
            ci_upper.append(d['mean'] + ci_high)
        
        plt.fill_between(epochs, ci_lower, ci_upper,
                        alpha=0.3, color='blue', 
                        label=f'{int(confidence*100)}% Confidence Interval')
    
    plt.xlabel('Epoch')
    plt.ylabel('Caption Score')
    plt.title('Caption Score vs Epoch (Averaged Across Runs)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add info about number of runs
    num_runs = max(d['count'] for d in aggregated_data)
    min_runs = min(d['count'] for d in aggregated_data)
    if num_runs == min_runs:
        plt.figtext(0.02, 0.02, f'Data from {num_runs} runs', fontsize=8)
    else:
        plt.figtext(0.02, 0.02, f'Data from {min_runs}-{num_runs} runs per epoch', fontsize=8)
    
    plt.tight_layout()
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(
        description='Plot averaged caption scores across multiple runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s SMB1-conditional-MiniLM-regular 0-4
  %(prog)s run 0,1,2,3,4,5 --std
  %(prog)s experiment 1-10 --ci --confidence 0.99
  %(prog)s my_run 0,2,4,6,8 --ci --output results.png
        """
    )
    
    parser.add_argument('prefix', 
                       help='Directory name prefix (e.g., "SMB1-conditional-MiniLM-regular")')
    parser.add_argument('run_ids', 
                       help='Run IDs to include. Format: "0-4" for range or "0,1,2,3" for list')
    parser.add_argument('--std', action='store_true',
                       help='Show standard deviation as error region')
    parser.add_argument('--ci', action='store_true',
                       help='Show confidence intervals as error region')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level for CI (default: 0.95)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output filename (default: show plot)')
    
    args = parser.parse_args()
    
    # Parse run IDs
    try:
        if '-' in args.run_ids:
            start, end = map(int, args.run_ids.split('-'))
            run_ids = list(range(start, end + 1))
        else:
            run_ids = [int(x.strip()) for x in args.run_ids.split(',')]
    except ValueError:
        print(f"Error: Invalid run_ids format '{args.run_ids}'. Use 'start-end' or 'id1,id2,id3'")
        sys.exit(1)
    
    print(f"Looking for runs: {run_ids}")
    print(f"Directory prefix: {args.prefix}")
    
    # Collect data from all runs
    all_data = collect_run_data(args.prefix, run_ids)
    
    # Aggregate data by epoch
    aggregated = aggregate_data(all_data)
    print(f"Aggregated data for {len(aggregated)} epochs")
    
    # Determine error type
    error_type = None
    if args.std and args.ci:
        print("Error: Cannot use both --std and --ci. Choose one.")
        sys.exit(1)
    elif args.std:
        error_type = 'std'
    elif args.ci:
        error_type = 'ci'
    
    # Create plot
    fig = create_plot(aggregated, error_type, args.confidence)
    
    # Save or show plot
    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
