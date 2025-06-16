import argparse
import itertools
import os
import random

import os

import util.common_settings as common_settings  # adjust import if needed
from level_dataset import LevelDataset  # adjust import if needed
from torch.utils.data import DataLoader
from evaluate_caption_adherence import calculate_caption_score_and_samples  # adjust import if needed
from verify_data_complete import detect_caption_order_tolerance, find_last_line_caption_order_tolerance
#import matplotlib.pyplot as plt
#import matplotlib
import json
#from tqdm import tqdm
import re

import numpy as np
import torch
#from tqdm import tqdm

from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from captions.util import extract_tileset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption order tolerance for a diffusion model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--caption", type=str, required=False, default=None, help="Caption to evaluate, phrases separated by periods")
    parser.add_argument("--tileset", type=str, help="Path to the tileset JSON file")
    #parser.add_argument("--json", type=str, default="datasets\\Test_for_caption_order_tolerance.json", help="Path to dataset json file")
    #parser.add_argument("--json", type=str, default="datasets\\SMB1_LevelsAndCaptions-regular-test.json", help="Path to dataset json file")
    parser.add_argument("--json", type=str, default="datasets\\Mar1and2_LevelsAndCaptions-regular.json", help="Path to dataset json file")
    #parser.add_argument("--trials", type=int, default=3, help="Number of times to evaluate each caption permutation")
    parser.add_argument("--inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=common_settings.GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--game", type=str, choices=["Mario", "LR"], default="Mario", help="Game to evaluate (Mario or Lode Runner)")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory if not comparing checkpoints (subdir of model directory)")
    parser.add_argument("--max_permutations", type=int, default=5, help="Maximum amount of permutations that can be made per caption")
    parser.add_argument("--start_line", type=int, default=0, help="Where in the jsonl file should it continue from")
    return parser.parse_args()


def setup_environment(seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return device

def load_captions_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # If the JSON is a list of dicts with a "caption" key
    captions = [entry["caption"] for entry in data if "caption" in entry]
    return captions

def creation_of_parameters(caption, max_permutations):
    args = parse_args() # Do parse_args only once, but make args global if that helps
    device = setup_environment(args.seed) # Just once

    # This also does not belong here
    if args.game == "Mario":
        num_tiles = common_settings.MARIO_TILE_COUNT
        tileset = '..\TheVGLC\Super Mario Bros\smb.json'
    elif args.game == "LR":
        num_tiles = common_settings.LR_TILE_COUNT
        tileset = '..\TheVGLC\Lode Runner\Loderunner.json'
    else:
        raise ValueError(f"Unknown game: {args.game}")

    # Load pipeline
    pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to(device)

    # Load tile metadata
    # Just once
    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset)

    perm_captions = []
    if isinstance(caption, list): # get rid of?
        # captions is a list of caption strings
        phrases_per_caption = [
            [p.strip() for p in cap.split('.') if p.strip()]
            for cap in caption
        ]
        permutations = []
        for phrases in phrases_per_caption:
            perms = list(itertools.permutations(phrases))
            if len(perms) > max_permutations:
                perms = random.sample(perms, max_permutations)
            permutations.append(perms)
        perm_captions = ['.'.join(perm) + '.' for perms in permutations for perm in perms]
    elif isinstance(caption, str):
        # Split caption into phrases and get all permutations
        phrases = [p.strip() for p in caption.split('.') if p.strip()]
        #print("phrase: ", phrases)
        permutations_cap = []
        perms = get_random_permutations(phrases, max_permutations)

        perm_captions = ['.'.join(perm) + '.' for perm in perms]

     # Create a list of dicts as expected by LevelDataset
    caption_data = [{"scene": None, "caption": cap} for cap in perm_captions]

    # Initialize dataset
    dataset = LevelDataset(
        data_as_list=caption_data,
        shuffle=False,
        mode="text",
        augment=False,
        num_tiles=common_settings.MARIO_TILE_COUNT,
        negative_captions=False,
        block_embeddings=None
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=min(16, len(perm_captions)),
        shuffle=False,
        num_workers=4,
        drop_last=False,
        persistent_workers=True
    )


    return pipe, device, id_to_char, char_to_id, tile_descriptors, num_tiles, dataloader, perm_captions, caption_data

def statistics_of_captions(captions, dataloader, compare_all_scores, count, pipe=None, device=None, id_to_char=None, char_to_id=None, tile_descriptors=None, num_tiles=None):
    """
    Calculate statistics of the captions.
    Returns average, standard deviation, minimum, maximum, and median of caption scores.
    """
    args = parse_args()
    if not captions:
        print("No captions found in the provided JSON file.")
        return
    print(f"\nLoaded {len(captions)} captions from {args.json}")

    
    avg_score = np.mean(compare_all_scores)
    std_dev_score = np.std(compare_all_scores)
    min_score = np.min(compare_all_scores)
    max_score = np.max(compare_all_scores)
    median_score = np.median(compare_all_scores)
    
    print("\n-----Scores for each caption permutation-----")
    for i, score in enumerate(compare_all_scores):
        print(f"Scores for caption {i + 1}:", score)
    caption_variable = "\n-----Statistics of caption " + str(count) +"-----"
    print(caption_variable)
    print(f"Average score: {avg_score:.4f}")
    print(f"Standard deviation: {std_dev_score:.4f}")
    print(f"Minimum score: {min_score:.4f}")
    print(f"Maximum score: {max_score:.4f}")
    print(f"Median score: {median_score:.4f}")

    return compare_all_scores, avg_score, std_dev_score, min_score, max_score, median_score

def get_random_permutations(phrases, max_permutations):
    seen = set()
    results = []
    attempts = 0
    max_attempts = max_permutations * 10  # avoid infinite loop in small cases

    while len(results) < max_permutations and attempts < max_attempts:
        perm = tuple(random.sample(phrases, len(phrases)))  # random permutation
        if perm not in seen:
            seen.add(perm)
            results.append(perm)
        attempts += 1

    return results

def get_old_captions(output_jsonl_path, file, all_scores, all_avg_scores, all_std_dev_scores, all_min_scores, all_max_scores, all_median_scores, counts):
    if file is None or not os.path.exists(output_jsonl_path):
        return all_scores, all_avg_scores, all_std_dev_scores, all_min_scores, all_max_scores, all_median_scores, counts
    with open(output_jsonl_path, "r") as f:
        print("output_jsonl_path:", output_jsonl_path)
        print("file:", file)
        # quit()
        for line in f:
            data = json.loads(line)

             # Find the key that starts with "Caption"
            caption_text = next((v for k, v in data.items() if k.startswith("Caption")), "")
            
            # Split on punctuation like . or ; and remove empty parts
            phrases = [p.strip() for p in re.split(r"[.;]", caption_text) if p.strip()]
            counts.append(len(phrases))

            # Append values to each list
            all_avg_scores.append(data.get("Average score for all permutations", 0))
            all_std_dev_scores.append(data.get("Standard deviation", 0))
            all_min_scores.append(data.get("Minimum score", 0))
            all_max_scores.append(data.get("Maximum score", 0))
            all_median_scores.append(data.get("Median score", 0))

    return all_scores, all_avg_scores, all_std_dev_scores, all_min_scores, all_max_scores, all_median_scores, counts


def main():
    args = parse_args()
    if args.caption is None or args.caption == "":
        caption = load_captions_from_json(args.json)
    else:
        caption = args.caption
        #caption = ("many pipes. many coins. , many enemies. many blocks. , many platforms. many question blocks.").split(',')
    phrases_per_model_path = [p.strip() for p in args.model_path.split('\\') if p.strip()]
    model_name = phrases_per_model_path[-1]
    phrases_per_json_path = [p.strip() for p in re.split(r'[\\.]', args.json) if p.strip()]
    json_name = phrases_per_json_path[-2]

    all_scores = []
    all_avg_scores = []
    all_std_dev_scores = []
    all_min_scores = []
    all_max_scores = []
    all_median_scores = []
    just_all_scores = []
    num_captions = []
    all_captions =  [item.strip() for s in caption for item in s.split(",")]

    one_caption = []
    total_num_perms = 0

    _, file = detect_caption_order_tolerance(args.model_path)

    if file is None and args.start_line == 0:
        letter = "w"
        count = 1
    else:
        letter = "a"
        # if last_line <= 0 then the file is empty
        # else it is the number of the last caption in the jsonl file
        last_line = find_last_line_caption_order_tolerance(args.model_path, file, key="Caption")
        #all_scores, all_avg_scores, all_std_dev_scores, all_min_scores, all_max_scores, all_median_scores, num_captions = get_old_captions(output_jsonl_path, file, all_scores, all_avg_scores, all_std_dev_scores, all_min_scores, all_max_scores, all_median_scores, num_captions)
    
        count = last_line + 1

    file_name = json_name + '_caption_order_tolerance.jsonl'
    #os.makedirs(folder_name,  exist_ok=True)
    output_jsonl_path = os.path.join(args.model_path, file_name)

    all_scores, all_avg_scores, all_std_dev_scores, all_min_scores, all_max_scores, all_median_scores, num_captions = get_old_captions(output_jsonl_path, file, all_scores, all_avg_scores, all_std_dev_scores, all_min_scores, all_max_scores, all_median_scores, num_captions)
    
    # Adds old permuations to total amount of permutations
    for num in num_captions:
            if num == 1:
                total_num_perms += 1
            elif num == 2:
                total_num_perms += 2
            else:
                total_num_perms += 5


    with open(output_jsonl_path, letter) as f:
        while(count <= len(all_captions)):
        #for cap in all_captions:
            one_caption = all_captions[count - 1]
            # print("length of caption:", len(one_caption))
            # print("caption:", one_caption)
            # Determines the amount of permutations for a caption

            # Initialize dataset
            pipe, device, id_to_char, char_to_id, tile_descriptors, num_tiles, dataloader, perm_caption, caption_data = creation_of_parameters(one_caption, args.max_permutations)
            if not pipe:
                print("Failed to create pipeline.")
                return
            # print("perm_caption:", perm_caption)
            if len(perm_caption) == 1:
                num_perms = 1
            elif len(perm_caption) == 2:
                num_perms = 2
            else:
                num_perms = 5




            avg_score, all_samples, all_prompts, compare_all_scores = calculate_caption_score_and_samples(device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=True, height=common_settings.MARIO_HEIGHT, width=common_settings.MARIO_WIDTH)
            scores, avg_score, std_dev_score, min_score, max_score, median_score = statistics_of_captions(perm_caption, dataloader, compare_all_scores, count, pipe, device, id_to_char, char_to_id, tile_descriptors, num_tiles)
            caption_variable = 'Caption ' + str(count)
            if args.save_as_json:
                result_entry = {
                        caption_variable: one_caption,
                        "Average score for all permutations": avg_score,
                        "Standard deviation": std_dev_score,
                        "Minimum score": min_score,
                        "Maximum score": max_score,
                        "Median score": median_score,
                        "Number of permutations": num_perms
                    }
                f.write(json.dumps(result_entry) + "\n") 

            all_avg_scores.append(avg_score)
        
            for score in enumerate(scores):
                all_scores.append(score)
                just_all_scores.append(score[1]) 
            all_std_dev_scores.append(std_dev_score)
            all_min_scores.append(min_score)
            all_max_scores.append(max_score)
            all_median_scores.append(median_score)
            if (count % 10) == 0:
                f.flush()  # Ensure each result is written immediately
                os.fsync(f.fileno())  # Ensure file is flushed to disk
            count = count + 1
            total_num_perms += num_perms
        
        # all_avg_score = np.mean(all_avg_scores)
        # all_std_dev_score = np.std(all_std_dev_scores)
        # all_min_score = np.min(all_min_scores)
        # all_max_score = np.max(all_max_scores)
        # all_median_score = np.median(all_median_scores)
        # num_captions = len(all_scores)

        # get the stats only of the averages
        all_avg_score = np.mean(all_avg_scores)
        all_std_dev_score = np.std(all_avg_scores)
        all_min_score = np.min(all_avg_scores)
        all_max_score = np.max(all_avg_scores)
        all_median_score = np.median(all_avg_scores)

        results = {

                "Scores of all captions": {
                    "Number of captions": count - 1,
                    "Average of average permutations": all_avg_score,
                    "Average of average permutations": np.mean(just_all_scores),
                    "Standard deviation of average permutations": all_std_dev_score,
                    "Min score of average permutations": all_min_score,
                    "Max score of average permutations": all_max_score,
                    "Median score of average permutations": all_median_score
                },
            }
        json.dump(results, f, indent=4)

        print(f"Results saved to {output_jsonl_path}")

    print("\nTotal number of permutations:", total_num_perms)
    print(f"\nAverage score across all captions: {avg_score:.4f}")

    print("\nAll samples shape:", all_samples.shape)
  
if __name__ == "__main__":
    main()
