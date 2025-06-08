import argparse
import json
import glob
import os
from datetime import datetime

def find_training_log(model_path):
    # Look for files matching the training log pattern
    log_files = glob.glob(os.path.join(model_path, "training_log_*.jsonl"))
    
    if not log_files:
        raise FileNotFoundError(f"No training log files found in {model_path}")
    if len(log_files) > 1:
        raise ValueError(f"Multiple training log files found in {model_path}: {log_files}")
    
    return log_files[0]

def get_timestamps(log_file):
    # Get first entry
    with open(log_file, 'r') as f:
        first_entry = json.loads(f.readline().strip())
        
        # Read through file to get to the last entry
        last_entry = None
        for line in f:
            if line.strip():  # Skip empty lines
                last_entry = json.loads(line.strip())
    
    if not last_entry:
        raise ValueError(f"Training log file {log_file} appears to be empty or malformed")
    
    return first_entry['timestamp'], last_entry['timestamp']

def calculate_duration(start_time, end_time):
    # Parse timestamps and calculate difference
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    
    duration = end_dt - start_dt
    return duration

def main():
    parser = argparse.ArgumentParser(description="Calculate model training execution time")
    parser.add_argument("model_path", help="Path to the model directory")
    
    args = parser.parse_args()
    
    try:
        log_file = find_training_log(args.model_path)
        start_time, end_time = get_timestamps(log_file)
        duration = calculate_duration(start_time, end_time)
        
        print(f"Training started at: {start_time}")
        print(f"Training ended at:   {end_time}")
        print(f"Total training time: {duration}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()