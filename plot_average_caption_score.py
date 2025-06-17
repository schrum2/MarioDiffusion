#!/usr/bin/env python3
"""
Script to plot averaged caption scores across multiple runs and experiment batches.
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
import matplotlib.colors as mcolors

def find_jsonl_file(directory, file_pattern="caption_score_log_*.jsonl"):
    """Find the JSONL file matching the pattern in the given directory."""
    pattern = os.path.join(directory, file_pattern)
    files = glob.glob(pattern)
    if not files:
        return None
    if len(files) > 1:
        print(f"Warning: Multiple JSONL files found in {directory} matching '{file_pattern}', using {files[0]}")
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

def collect_run_data(prefix, run_ids, file_pattern="caption_score_log_*.jsonl"):
    """Collect data from all specified runs for a single experiment batch."""
    all_data = {}
    
    for run_id in run_ids:
        dir_name = f"{prefix}{run_id}"
        if not os.path.exists(dir_name):
            print(f"Warning: Directory {dir_name} not found, skipping...")
            continue
            
        jsonl_file = find_jsonl_file(dir_name, file_pattern)
        if not jsonl_file:
            print(f"Warning: No JSONL file matching '{file_pattern}' found in {dir_name}, skipping...")
            continue
            
        df = read_jsonl_data(jsonl_file)
        if df is None or df.empty:
            print(f"Warning: No data found in {jsonl_file}, skipping...")
            continue
            
        all_data[run_id] = df
    
    if all_data:
        print(f"Loaded data from {len(all_data)} runs for prefix '{prefix}' (pattern: {file_pattern})")
    else:
        print(f"Warning: No valid data found for prefix '{prefix}' with pattern '{file_pattern}'")
        
    return all_data

def aggregate_data(all_data, require_all_runs=True):
    """Aggregate data across runs by epoch."""
    if not all_data:
        return []
    
    num_runs = len(all_data)
    
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
                score = epoch_data.iloc[-1]['score']
                scores.append(score)
        
        # Determine whether to include this epoch
        include_epoch = False
        if require_all_runs:
            # Only include if present in ALL runs
            include_epoch = len(scores) == num_runs
        else:
            # Include if present in at least one run
            include_epoch = len(scores) > 0
        
        if include_epoch:
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

def parse_experiment_spec(spec, default_pattern="caption_score_log_*.jsonl"):
    """Parse experiment specification in format 'prefix:run_ids' or 'prefix:run_ids:file_pattern' or 'prefix:run_ids:file_pattern:label'."""
    parts = spec.split(':')
    if len(parts) < 2:
        raise ValueError(f"Invalid experiment spec '{spec}'. Format should be 'prefix:run_ids' or 'prefix:run_ids:file_pattern' or 'prefix:run_ids:file_pattern:label'")
    
    prefix = parts[0]
    run_ids_str = parts[1]
    file_pattern = parts[2] if len(parts) > 2 else default_pattern
    label = parts[3] if len(parts) > 3 else None
    
    # Parse run IDs
    try:
        if '-' in run_ids_str:
            start, end = map(int, run_ids_str.split('-'))
            run_ids = list(range(start, end + 1))
        else:
            run_ids = [int(x.strip()) for x in run_ids_str.split(',')]
    except ValueError:
        raise ValueError(f"Invalid run_ids format '{run_ids_str}'. Use 'start-end' or 'id1,id2,id3'")
    
    return prefix, run_ids, file_pattern, label

def get_color_palette(n):
    """Get a color palette with n distinct colors."""
    if n <= 10:
        # Use tab10 colormap for up to 10 colors
        return plt.cm.tab10(np.linspace(0, 1, n))
    else:
        # Use a combination of colormaps for more colors
        colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
        colors2 = plt.cm.Set3(np.linspace(0, 1, n - 10))
        return np.vstack([colors1, colors2])

def get_line_styles(n):
    """Get distinct line styles and markers for n different lines (black-and-white friendly)."""
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    
    styles = []
    for i in range(n):
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        styles.append((line_style, marker))
    
    return styles

def create_plot(experiment_data, error_type=None, confidence=0.95, title=None, legend_loc='lower right'):
    """Create the matplotlib plot with multiple experiment batches."""
    plt.figure(figsize=(12, 8))
    
    # Get colors and line styles for each experiment
    colors = get_color_palette(len(experiment_data))
    line_styles = get_line_styles(len(experiment_data))
    
    # Track all data for consistent axis limits
    all_epochs = []
    all_scores = []
    
    # Plot each experiment batch
    for i, (exp_name, aggregated_data) in enumerate(experiment_data.items()):
        if not aggregated_data:
            print(f"Warning: No data to plot for experiment '{exp_name}'")
            continue
            
        epochs = [d['epoch'] for d in aggregated_data]
        means = [d['mean'] for d in aggregated_data]
        color = colors[i]
        line_style, marker = line_styles[i]
        
        all_epochs.extend(epochs)
        all_scores.extend(means)
        
        # Main line plot with distinct styles for black-and-white compatibility
        plt.plot(epochs, means, color=color, linewidth=2.5, label=exp_name, 
                linestyle=line_style, marker=marker, markersize=6, markevery=1)
        
        # Add error regions if requested
        if error_type == 'std':
            stds = [d['std'] for d in aggregated_data]
            plt.fill_between(epochs, 
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.15, color=color)
        
        elif error_type == 'ci':
            ci_lower = []
            ci_upper = []
            for d in aggregated_data:
                ci_low, ci_high = calculate_confidence_interval(d['scores'], confidence)
                ci_lower.append(d['mean'] + ci_low)
                ci_upper.append(d['mean'] + ci_high)
            
            plt.fill_between(epochs, ci_lower, ci_upper,
                            alpha=0.15, color=color)
    
    # Formatting
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Caption Score', fontsize=18)
    
    # Set title if provided
    # if title is not None:
    #     plt.title(title, fontsize=14)
    # else:
    #     plt.title('Caption Score vs Epoch (Averaged Across Runs)', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    
    # Legend with specified location
    if legend_loc.startswith('bbox_to_anchor'):
        # Handle outside legend placement
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 16}, handlelength=2.5, handletextpad=1.5)
    else:
        legend = plt.legend(loc=legend_loc, prop={'size': 16}, handlelength=2.5, handletextpad=1.5, ncol=2)
    
    # Add error type to legend title if applicable
    # if error_type == 'std':
        # legend.set_title('Experiments (± 1 std)', prop={'size': 20})
    # elif error_type == 'ci':
        # legend.set_title(f'Experiments ({int(confidence*100)}% CI)', prop={'size': 20})
    # else:
        # legend.set_title('Experiments', prop={'size': 20})
    
    # Add info about runs at bottom
    info_lines = []
    for exp_name, aggregated_data in experiment_data.items():
        if aggregated_data:
            num_runs = max(d['count'] for d in aggregated_data)
            min_runs = min(d['count'] for d in aggregated_data)
            if num_runs == min_runs:
                info_lines.append(f"{exp_name}: {num_runs} runs")
            else:
                info_lines.append(f"{exp_name}: {min_runs}-{num_runs} runs")
    
    #if info_lines:
    #    plt.figtext(0.02, 0.02, ' | '.join(info_lines), fontsize=8, wrap=True)
    
    plt.tight_layout()
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(
        description='Plot averaged caption scores across multiple runs and experiment batches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment batch (default pattern)
  %(prog)s "SMB1-conditional-MiniLM-regular:0-4"
  
  # Multiple experiment batches with default pattern
  %(prog)s "experiment1:0-4" "experiment2:0-4" "baseline:0,2,4"
  
  # Compare same experiments with different file patterns
  %(prog)s "SMB1-regular:0-4:SMB1_LevelsAndCaptions-*_scores_by_epoch.jsonl" "SMB1-regular:0-4:SMB1_ValidationCaptions-*_scores_by_epoch.jsonl"
  
  # Mix of default and custom patterns
  %(prog)s "SMB1-regular:0-4" "SMB1-regular:0-4:*ValidationCaptions*_scores_by_epoch.jsonl"
  
  # Using global file pattern (applies to all experiments without specific pattern)
  %(prog)s "SMB1-regular:0-4" "experiment2:0-4" -f "*_scores_by_epoch.jsonl"
  
  # Custom labels and title
  %(prog)s "exp1:0-3::Training Data" "exp1:0-3:validation_*.jsonl:Validation Data" --title "Training vs Validation Performance"
  
  # No title and custom legend location
  %(prog)s "SMB1-regular:0-4" --title none --legend-loc "upper left"
  
  # With error bars comparing training vs validation (only epochs in all runs)
  %(prog)s "SMB1-regular:0-4:*LevelsAndCaptions*_scores_by_epoch.jsonl" "SMB1-regular:0-4:*ValidationCaptions*_scores_by_epoch.jsonl" --std
  
  # Include partial epochs (epochs not present in all runs)
  %(prog)s "SMB1-regular:0-4" --allow-partial
  
  # Save comparison plot
  %(prog)s "run1:0-4:pattern1.jsonl" "run1:0-4:pattern2.jsonl" --output comparison.png

  # Actual example
  %(prog)s "SMB1-conditional-MiniLM-regular:0-3:SMB1_RandomTest-regular_scores_by_epoch.jsonl:Random Captions" "SMB1-conditional-MiniLM-regular:0-3:SMB1_LevelsAndCaptions-regular_scores_by_epoch.jsonl:Real Captions" --title "Caption Score Using MiniLM"

Format for experiment specification: 
  - "prefix:run_ids" (uses global --file-pattern)
  - "prefix:run_ids:file_pattern" (uses specific pattern)
  - "prefix:run_ids:file_pattern:label" (uses specific pattern and custom label)
  
  Where:
  - prefix: Directory name prefix
  - run_ids: "start-end" for range or "id1,id2,id3" for list
  - file_pattern: File pattern with wildcards (optional)
  - label: Custom label for legend (optional)

File patterns support wildcards:
  - "caption_score_log_*.jsonl" (default)
  - "*_scores_by_epoch.jsonl"
  - "SMB1_LevelsAndCaptions-*_scores_by_epoch.jsonl"
  - "SMB1_ValidationCaptions-*_scores_by_epoch.jsonl"
        """
    )
    
    parser.add_argument('experiments', nargs='+',
                       help='Experiment specifications in format "prefix:run_ids"')
    parser.add_argument('--file-pattern', '-f', type=str, default="caption_score_log_*.jsonl",
                       help='Default file pattern for experiments without specific pattern (default: "caption_score_log_*.jsonl")')
    # parser.add_argument('--title', type=str, default='Caption Score vs Epoch (Averaged Across Runs)',
    #                    help='Title for the plot (use "none" for no title)')
    parser.add_argument('--legend-loc', type=str, default='lower right',
                       help='Legend location (e.g., "lower right", "upper left", "center", "outside")')
    parser.add_argument('--allow-partial', action='store_true',
                       help='Include epochs that are not present in all runs (default: only plot epochs present in all runs)')
    parser.add_argument('--std', action='store_true',
                       help='Show standard deviation as error region')
    parser.add_argument('--ci', action='store_true',
                       help='Show confidence intervals as error region')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level for CI (default: 0.95)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output filename (default: show plot)')
    
    args = parser.parse_args()
    
    # Validate error type arguments
    if args.std and args.ci:
        print("Error: Cannot use both --std and --ci. Choose one.")
        sys.exit(1)
    
    # Parse experiment specifications
    experiment_configs = {}
    try:
        for i, exp_spec in enumerate(args.experiments):
            prefix, run_ids, file_pattern, label = parse_experiment_spec(exp_spec, args.file_pattern)
            
            # Create experiment name - use custom label if provided
            if label:
                exp_name = label
            elif file_pattern != args.file_pattern:
                # Use a more descriptive name based on the file pattern
                pattern_key = file_pattern.replace('*', '').replace('.jsonl', '').replace('_scores_by_epoch', '')
                pattern_key = pattern_key.replace('SMB1_', '').replace('-regular', '')
                if pattern_key:
                    exp_name = f"{prefix}:{pattern_key}"
                else:
                    exp_name = f"{prefix}:exp{i+1}"
            else:
                exp_name = prefix
            
            # Handle duplicate experiment names by adding suffix
            original_name = exp_name
            counter = 1
            while exp_name in experiment_configs:
                exp_name = f"{original_name}_{counter}"
                counter += 1
            
            experiment_configs[exp_name] = (prefix, run_ids, file_pattern)
            print(f"Experiment '{exp_name}': prefix '{prefix}', runs {run_ids}, pattern '{file_pattern}'")
    except ValueError as e:
       print(f"Error: {e}")
       sys.exit(1)
    
    # Collect and aggregate data for each experiment batch
    experiment_data = {}
    for exp_name, (prefix, run_ids, file_pattern) in experiment_configs.items():
        all_data = collect_run_data(prefix, run_ids, file_pattern)
        aggregated = aggregate_data(all_data, require_all_runs=not args.allow_partial)
        experiment_data[exp_name] = aggregated
    
    # Check if we have any valid data
    if not any(experiment_data.values()):
        print("Error: No valid data found for any experiments!")
        sys.exit(1)
    
    # Determine error type
    error_type = None
    if args.std:
        error_type = 'std'
    elif args.ci:
        error_type = 'ci'
    
    # Handle title and legend location
    #title = None if args.title.lower() == 'none' else args.title
    legend_loc = 'bbox_to_anchor=(1.05, 1), loc=upper left' if args.legend_loc == 'outside' else args.legend_loc
    
    # Create plot
    fig = create_plot(experiment_data, error_type, args.confidence, legend_loc)
    
    # Save or show plot
    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()