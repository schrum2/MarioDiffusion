import argparse
import json
import glob
import os
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple

def find_training_log(model_path):
    # Look for files matching the training log pattern
    log_files = glob.glob(os.path.join(model_path, "training_log_*.jsonl"))
    
    if not log_files:
        raise FileNotFoundError(f"No training log files found in {model_path}")
    if len(log_files) > 1:
        raise ValueError(f"Multiple training log files found in {model_path}: {log_files}")
    
    return log_files[0]

def get_timestamps(log_file, best_epoch=None):
    """
    Returns (start_time, end_time, best_epoch_time) where best_epoch_time is None if not requested.
    """
    best_epoch_time = None
    with open(log_file, 'r') as f:
        first_entry = json.loads(f.readline().strip())
        last_entry = None
        for line in f:
            if line.strip():
                entry = json.loads(line.strip())
                last_entry = entry
                if best_epoch is not None and best_epoch_time is None:
                    if entry.get("epoch") == best_epoch:
                        best_epoch_time = entry["timestamp"]
    if not last_entry:
        raise ValueError(f"Training log file {log_file} appears to be empty or malformed")
    return first_entry['timestamp'], last_entry['timestamp'], best_epoch_time

def calculate_duration(start_time, end_time):
    # Parse timestamps and calculate difference
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    
    duration = end_dt - start_dt
    return duration

def seconds_to_time_str(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def calculate_statistics(durations: List[float]) -> Dict:
    """Calculate various statistics from a list of durations in seconds"""
    duration_array = np.array(durations)
      # Calculate raw statistics
    q1, q3 = float(np.percentile(duration_array, 25)), float(np.percentile(duration_array, 75))
    stats = {
        "raw": {
            "mean": float(np.mean(duration_array)),
            "std": float(np.std(duration_array)),
            "stderr": float(np.std(duration_array) / np.sqrt(len(duration_array))),
            "median": float(np.median(duration_array)),
            "q1": q1,
            "q3": q3,
            "min": float(np.min(duration_array)),
            "max": float(np.max(duration_array)),
            "individual_times": durations
        }
    }
    
    # Add formatted times
    stats["formatted"] = {
        "mean": seconds_to_time_str(stats["raw"]["mean"]),
        "std": seconds_to_time_str(stats["raw"]["std"]),
        "stderr": seconds_to_time_str(stats["raw"]["stderr"]),
        "median": seconds_to_time_str(stats["raw"]["median"]),
        "q1": seconds_to_time_str(q1),
        "q3": seconds_to_time_str(q3),
        "min": seconds_to_time_str(stats["raw"]["min"]),
        "max": seconds_to_time_str(stats["raw"]["max"]),
        "individual_times": [seconds_to_time_str(d) for d in durations]
    }
    
    return stats

def save_statistics(stats: Dict, prefix: str):
    """Save statistics to a JSON file"""
    output_file = f"{prefix}-runtime-plus-best.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Calculate model training execution time")
    parser.add_argument("prefix", help="Prefix of the model path")
    parser.add_argument("start_num", type=int, help="Start number for model runs")
    parser.add_argument("end_num", type=int, help="End number for model runs (inclusive)")
    
    args = parser.parse_args()
    
    durations_seconds = []
    best_epoch_durations_seconds = []
    failed_runs = []
    
    for run_num in range(args.start_num, args.end_num + 1):
        model_path = f"{args.prefix}{run_num}"
        print(f"\nAnalyzing run {run_num}:")
        
        try:
            log_file = find_training_log(model_path)
            # Load best_epoch from best_model_info.json
            best_model_info_path = os.path.join(model_path, "best_model_info.json")
            if not os.path.exists(best_model_info_path):
                raise FileNotFoundError(f"No best_model_info.json found in {model_path}")
            with open(best_model_info_path, 'r') as f:
                best_model_info = json.load(f)
            best_epoch = best_model_info.get("best_epoch")
            start_time, end_time, best_epoch_time = get_timestamps(log_file, best_epoch=best_epoch)
            duration = calculate_duration(start_time, end_time)
            if best_epoch_time is not None:
                duration_to_best = calculate_duration(start_time, best_epoch_time)
            else:
                duration_to_best = None
            print(f"  Training started at: {start_time}")
            print(f"  Training ended at:   {end_time}")
            print(f"  Training time:       {duration}")
            if duration_to_best is not None:
                print(f"  Time to best epoch ({best_epoch}): {duration_to_best}")
            durations_seconds.append(duration.total_seconds())
            best_epoch_durations_seconds.append(duration_to_best.total_seconds() if duration_to_best is not None else None)
        except Exception as e:
            print(f"  Error: {str(e)}")
            failed_runs.append(run_num)
            continue
    
    if not durations_seconds:
        print("\nNo successful runs found.")
        exit(1)
    
    # Calculate and save statistics
    # Filter out None values for best_epoch_durations_seconds
    filtered_best_epoch_durations = [d for d in best_epoch_durations_seconds if d is not None]
    stats = {
        "overall_runtime": calculate_statistics(durations_seconds),
        "time_to_best_epoch": calculate_statistics(filtered_best_epoch_durations) if filtered_best_epoch_durations else None
    }
    output_file = save_statistics(stats, args.prefix)
    # Print summary
    print("\nSummary Statistics:")
    print(f"  Average runtime: {stats['overall_runtime']['formatted']['mean']}")
    print(f"  Average time to best epoch: {stats['time_to_best_epoch']['formatted']['mean'] if stats['time_to_best_epoch'] else 'N/A'}")
    print(f"  Std deviation:  {stats['overall_runtime']['formatted']['std']}")
    print(f"  Std error:     {stats['overall_runtime']['formatted']['stderr']}")
    print(f"  Q1 (25th):     {stats['overall_runtime']['formatted']['q1']}")
    print(f"  Median:        {stats['overall_runtime']['formatted']['median']}")
    print(f"  Q3 (75th):     {stats['overall_runtime']['formatted']['q3']}")
    print(f"  Min:           {stats['overall_runtime']['formatted']['min']}")
    print(f"  Max:           {stats['overall_runtime']['formatted']['max']}")
    print(f"\nStatistics saved to: {output_file}")
    
    if failed_runs:
        print(f"\nFailed runs: {failed_runs}")

if __name__ == "__main__":
    main()