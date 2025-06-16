import os
import json
import argparse
import re
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Find the epoch of the best caption score for fdm runs")
    parser.add_argument("--model_path", type=str, default="temp-fdm-caption-score-testing\Mar1and2-fdm-MiniLM-absence3", help="Path to the directory containing the FDM runs.")
    #parser.add_argument("--random_test", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    epoch, score = find_best_score(args.model_path)
    print(f"The best epoch found was epoch {epoch} with a score of {score}")

    #Move the old
    os.rename(os.path.join(args.model_path, "final-model"), os.path.join(args.model_path, "final-model-old"))
    os.rename(os.path.join(args.model_path, "samples-from-random-Mar1and2-captions"), os.path.join(args.model_path, "samples-from-random-Mar1and2-captions-old"))
    os.rename(os.path.join(args.model_path, "samples-from-real-Mar1and2-captions"), os.path.join(args.model_path, "samples-from-real-Mar1and2-old"))
    shutil.copytree(os.path.join(args.model_path, f"checkpoint-{epoch}"), os.path.join(args.model_path, "final-model"))


#Find the best epoch caption score
def find_best_score(dir_path):
    base_score=0
    base_epoch=0
    pattern=r"caption_score_log*"

    filepath=None
    for filename in os.listdir(dir_path):
        if re.match(pattern, filename):
            filepath=os.path.join(dir_path, filename)
    
    if filepath==None:
        print(f"pattern:{pattern}", f"path: {dir_path}")
        filepath="caption_score_log_20250605-184054.jsonl"
    
    data=[]
    with open(filepath, 'r') as file:
        for line in file:
            line_data = json.loads(line)
            #We need this check because we only save every 10th epoch
            if(line_data['epoch']%10==0):
                data.append(line_data)
    
    for item in data:
        value = item['caption_score']
        if value>base_score:
            base_score=value
            base_epoch=item['epoch']
    
    return base_epoch, base_score


if __name__ == "__main__":
    main()