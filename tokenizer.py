import json
import re
from collections import Counter
import pickle
import sys

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text):
        # Match words, numbers, periods, and commas as separate tokens
        tokens = re.findall(r'\w+|[.,]', text.lower())
        return tokens

    def build_vocab(self, dataset_path, min_freq=1):
        token_counter = Counter()

        with open(dataset_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                caption = entry['caption']
                tokens = self.tokenize(caption)
                token_counter.update(tokens)

        # Keep tokens that meet the min frequency
        tokens = [tok for tok, count in token_counter.items() if count >= min_freq]

        # Build vocab dictionaries
        self.vocab = {tok: idx for idx, tok in enumerate(sorted(tokens))}
        self.token_to_id = self.vocab
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)}")

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id.get(tok, -1) for tok in tokens]  # -1 for unknowns

    def decode(self, token_ids):
        return ' '.join(self.id_to_token.get(tok_id, '<UNK>') for tok_id in token_ids)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'vocab': self.vocab}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.token_to_id = self.vocab
            self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}

    def get_vocab(self):
        return sorted(self.vocab.keys())

if __name__ == "__main__":
    tokenizer = Tokenizer()

    if sys.argv[1] == "save":
        tokenizer.build_vocab('SMB1_LevelsAndCaptions.json')
        tokenizer.save('SMB1_Tokenizer.pkl')
    elif sys.argv[1] == "load":
        tokenizer.load('SMB1_Tokenizer.pkl')

    # Example usage
    #print(tokenizer.encode("floor with one gap. one enemy."))
    #print(tokenizer.get_vocab())