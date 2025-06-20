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
    total_captions = 0
    num_captions = 0

    for fname in os.listdir(directory):
        if fname.endswith("_caption_order_tolerance.jsonl"):
            path = os.path.join(directory, fname)
            with open(path, "r", encoding="utf-8") as f:
                file_name = os.path.basename(directory)
                group_num = re.search(r'(\d+)$', file_name)
                if group_num:
                    number = group_num.group(1)
                file_name = re.sub(r'\d+$', '', file_name)  # Remove trailing digits
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # skip lines that are not valid JSON
                    if not isinstance(data, dict):
                        continue  # skip lines that are not JSON objects
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
                output_jsonl_path = os.path.join(args.dir, f"caption_order_stats{number}.jsonl")
                with open(output_jsonl_path, "a") as f:
                    result_entry = {
                        "group": fname,
                        "Number of captions": num_captions,
                        "List of average scores": avg_scores
                    }
                    f.write(json.dumps(result_entry) + "\n") 

    if not avg_scores:
        print("No caption order tolerance data found.")
        return

    print(f"Total captions: {num_captions}")
    print(f"Average of averages: {np.mean(avg_scores):.4f}")
    print(f"Std of averages: {np.std(avg_scores):.4f}")
    print(f"Min of averages: {np.min(avg_scores):.4f}")
    print(f"Max of averages: {np.max(avg_scores):.4f}")
    print(f"Median of averages: {np.median(avg_scores):.4f}")

    return file_name, num_captions, avg_scores, std_scores, min_scores, max_scores, median_scores, number

def write_caption_order_stats_to_jsonl(file_name, total_captions, avg_of_avg, std_of_avg, min_of_avg, max_of_avg, median_of_avg, number):
    """
    Writes the caption order statistics to a JSONL file.
    """
    output_jsonl_path = os.path.join(args.dir, f"caption_order_stats{number}.jsonl")
    with open(f"caption_order_stats{number}.jsonl", "a") as f:
        result_entry = {
            "group": file_name,
            "Number of captions": total_captions,
            "Average of average permutations": np.mean(avg_of_avg),
            "Standard deviation of average permutations": np.std(avg_of_avg),
            "Min score of average permutations": np.min(avg_of_avg),
            "Max score of average permutations": np.max(avg_of_avg),
            "Median score of average permutations": np.median(avg_of_avg),
            "Lists of average scores": avg_of_avg,
            # "Lists of standard deviation scores": std_of_avg,
            # "Lists of minimum scores": min_of_avg,
            # "Lists of maximum scores": max_of_avg,
            # "Lists of median scores": median_of_avg
        }
        f.write(json.dumps(result_entry) + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Recreate caption order stats from JSONL files.")
    parser.add_argument("--dir", type=str, help="Directory containing caption order JSONL files.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    write_caption_order_stats_to_jsonl(*recreate_caption_order_stats_from_jsonl(args.dir))