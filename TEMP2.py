import json
from datetime import datetime
import numpy as np
from scipy import stats

log_path = []
log_name = []

log_path.append(r"G:\.shortcut-targets-by-id\1C1aB0CgC2ozojftq5eagcmSg65fS7ttg\SURF Artifacts\EXPERIMENTS\MarioDiffusion-FIX-SMB-DATA\Mar1and2-fdm-GTE-regular7\training_log_20250605-150814.jsonl")
log_name.append("Mar1and2-fdm-GTE-regular7")

log_path.append(r"G:\.shortcut-targets-by-id\1C1aB0CgC2ozojftq5eagcmSg65fS7ttg\SURF Artifacts\EXPERIMENTS\MarioDiffusion-FIX-SMB-DATA\Mar1and2-fdm-GTE-regular27\training_log_20250609-164155.jsonl")
log_name.append("Mar1and2-fdm-GTE-regular27")

for log_path, log_name in zip(log_path, log_name):
    print(f"Processing log: {log_name}")
    # Read all timestamps
    timestamps = []
    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            timestamps.append(datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S"))

    # Compute time differences (in seconds), skipping gaps >= 1 hour
    time_diffs = []
    stalls = []
    for prev, curr in zip(timestamps, timestamps[1:]):
        diff = (curr - prev).total_seconds()
        if diff < 3600:
            time_diffs.append(diff)
        else:
            stalls.append(diff)

    # Sort from greatest to least
    time_diffs_sorted = sorted(time_diffs, reverse=True)

    print(f"Times for {log_name}")
    print(time_diffs_sorted)
    print(f"Stalls for {log_name}")
    print(stalls)

    # Sum of time differences and stalls
    total_time = sum(time_diffs)
    total_stalls = sum(stalls)
    print(f"Original training time to be corrected for {log_name}: {total_time + total_stalls} seconds")

    # Calculating ne time to train
    new_train_time_total = total_time + (time_diffs[0] * len(stalls))
    print(f"New estimated training time for {log_name}: {new_train_time_total} seconds")
    print(f"New estimated training time for {log_name}: {new_train_time_total / 3600} hours")

ind_times = [3868.0, 3890.0, 3853.0, 3860.0, 4519.0, 4552.0, 4537.0, 4605.0, 4534.0, 4543.0, 4515.0, 4453.0, 
             4524.0, 4507.0, 4497.0, 4497.0, 4493.0, 4512.0, 4513.0, 4514.0, 4522.0, 4515.0, 4511.0, 4509.0, 
             4558.0, 4504.0, 4633.0, 4538.0, 4455.0, 4492.0]

arr = np.array(ind_times)
mean = np.mean(arr)
std = np.std(arr, ddof=1)
stderr = stats.sem(arr)
median = np.median(arr)
q1 = np.percentile(arr, 25)
q3 = np.percentile(arr, 75)
min_ = np.min(arr)
max_ = np.max(arr)

print(f'"mean": {mean},')
print(f'"std": {std},')
print(f'"stderr": {stderr},')
print(f'"median": {median},')
print(f'"q1": {q1},')
print(f'"q3": {q3},')
print(f'"min": {min_},')
print(f'"max": {max_},')