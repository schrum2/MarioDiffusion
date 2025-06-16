from evaluate_caption_order_tolerance import *
from verify_data_complete import detect_caption_order_tolerance, find_last_line_caption_order_tolerance
import argparse 
import os
import json
import numpy as np

def recreate_caption_order_stats_from_jsonl(directory):
    """
    Reads all *_caption_order_tolerance.jsonl files in the given directory,
    aggregates the average permutation scores, and prints summary statistics.
    """
    avg_scores = []
    std_scores = []
    min_scores = []
    max_scores = []
    median_scores = []
    num_captions = 0

    for fname in os.listdir(directory):
        if fname.endswith("_caption_order_tolerance.jsonl"):
            path = os.path.join(directory, fname)
            with open(path, "r", encoding="utf-8") as f:
                file_name = f
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # skip lines that are not valid JSON
                    avg = data.get("Average score for all permutations", None)
                    std = data.get("Standard deviation", None)
                    min_ = data.get("Minimum score", None)
                    max_ = data.get("Maximum score", None)
                    median = data.get("Median score", None)
                    if avg is not None:
                        avg_scores.append(avg)
                    if std is not None:
                        std_scores.append(std)
                    if min_ is not None:
                        min_scores.append(min_)
                    if max_ is not None:
                        max_scores.append(max_)
                    if median is not None:
                        median_scores.append(median)
                    num_captions += 1
                    total_captions = data.get("Number of captions", None)
                    avg_of_avg = data.get("Average of average permutations", None)
                    std_of_avg = data.get("Standard deviation of average permutations", None)
                    min_of_avg = data.get("Min score of average permutations", None)
                    max_of_avg = data.get("Max score of average permutations", None)
                    median_of_avg = data.get("Median score of average permutations", None)

    if not avg_scores:
        print("No caption order tolerance data found.")
        return

    print(f"Total captions: {num_captions}")
    print(f"Average of averages: {np.mean(avg_scores):.4f}")
    print(f"Std of averages: {np.std(avg_scores):.4f}")
    print(f"Min of averages: {np.min(avg_scores):.4f}")
    print(f"Max of averages: {np.max(avg_scores):.4f}")
    print(f"Median of averages: {np.median(avg_scores):.4f}")

    output_jsonl_path = os.path.join(args.dir, "caption_order_stats1.json")
    with open(output_jsonl_path, "a") as f:
        result_entry = {
                        "group": file_name,
                        "Number of captions": total_captions,
                        "Average of average permutations": avg_of_avg,
                        "Standard deviation of average permutations": std_of_avg,
                        "Min score of average permutations": min_of_avg,
                        "Max score of average permutations": max_of_avg,
                        "Median score of average permutations": median_of_avg
                    }
        f.write(json.dumps(result_entry) + "\n") 

def parse_args():
    parser = argparse.ArgumentParser(description="Recreate caption order stats from JSONL files.")
    parser.add_argument("--dir", type=str, help="Directory containing caption order JSONL files.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    recreate_caption_order_stats_from_jsonl(args.dir)