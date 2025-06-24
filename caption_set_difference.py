# Get captions from one dataset that are not in another

import json
import sys
from captions.caption_match import compare_captions

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} file1.json file2.json")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    captions1 = [item['caption'] for item in data1 if 'caption' in item]
    captions2 = [item['caption'] for item in data2 if 'caption' in item]

    for cap1 in captions1:
        best_score = -float('inf')
        best_cap2 = None
        for cap2 in captions2:
            score = compare_captions(cap1, cap2)
            if score > best_score:
                best_score = score
                best_cap2 = cap2
                if best_score == 1.0:
                    break

        if best_score < 1.0:
            print(f"Caption: {cap1}\nBest score: {best_score:.3f}\nBest match: {best_cap2}\n---")

if __name__ == "__main__":
    main()

