import json
import re
from collections import Counter
import pickle
import sys
import argparse

class Tokenizer:
    def __init__(self):
        self.special_tokens = ["[PAD]", "[MASK]"]
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text):
        # Match words, numbers, periods, and commas as separate tokens
        tokens = re.findall(r'\w+|[.,]|\[mask\]|\[pad\]', text.lower())
        # Restore MASK and PAD to all caps
        modified_list = []
        for s in tokens:
            modified_s = s.replace("[mask]", "[MASK]").replace("[pad]", "[PAD]")
            modified_list.append(modified_s)
        return modified_list

    def pad_sequence(self, tokens, length):
        """Pads tokenized sequences to length with a padding token (assumed to be '[PAD]')."""
        pad_token = self.token_to_id["[PAD]"]
        return tokens + [pad_token] * (length - len(tokens))

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

        # Ensure special tokens are always included
        all_tokens = self.special_tokens + sorted(tokens)
        
        # Build vocab dictionaries
        self.vocab = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.token_to_id = self.vocab
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}

        print(f"Vocabulary size: {len(self.vocab)}")

    def encode(self, text):
        tokens = self.tokenize(text)
        encoded = []
        for tok in tokens:
            if tok not in self.token_to_id:
                raise ValueError(f"Unknown token encountered: {tok} in {text}")
            encoded.append(self.token_to_id[tok])
        return encoded

    def encode_batch(self, texts, pad_to_length=None):
        """
        Encode a batch of texts into token IDs with padding to ensure uniform length.
    
        Args:
            texts (list): A list of strings to encode
            pad_to_length (int, optional): Length to pad all sequences to. If None,
                                          will pad to the length of the longest sequence.
        
        Returns:
            list: A list of lists, where each inner list contains the token IDs for a text
        """
        # Get the padding token ID
        pad_token = self.token_to_id["[PAD]"]
    
        # First encode all texts
        encoded_texts = []
        for text in texts:
            try:
                encoded = self.encode(text)
                encoded_texts.append(encoded)
            except ValueError as e:
                raise ValueError(f"Error encoding text: {text}. {str(e)}")
    
        # Determine padding length
        if pad_to_length is None:
            pad_to_length = max(len(seq) for seq in encoded_texts)
    
        # Pad sequences to uniform length
        padded_texts = []
        for seq in encoded_texts:
            if len(seq) > pad_to_length:
                # Truncate if too long
                padded_texts.append(seq[:pad_to_length])
            else:
                # Pad if too short
                padding = [pad_token] * (pad_to_length - len(seq))
                padded_texts.append(seq + padding)
    
        return padded_texts

    def decode(self, token_ids):
        return ' '.join(self.id_to_token[tok_id] for tok_id in token_ids)

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

    def get_vocab_size(self):
        return len(self.vocab)

if __name__ == "__main__":
    tokenizer = Tokenizer()

    parser = argparse.ArgumentParser(description="Tokenizer utility for saving and loading vocabularies.")
    parser.add_argument("action", choices=["save", "load"], help="Action to perform: 'save' or 'load'.")
    parser.add_argument("--json_file", type=str, default='SMB1_LevelsAndCaptions.json', help="Path to the JSON file containing the dataset (required for 'save').")
    parser.add_argument("--pkl_file", type=str, default='SMB1_Tokenizer.pkl', help="Path to the pickle file to save/load the tokenizer.")

    args = parser.parse_args()

    if args.action == "save":
        if not args.json_file:
            raise ValueError("The --json_file argument is required for the 'save' action.")
        tokenizer.build_vocab(args.json_file)
        tokenizer.save(args.pkl_file)
    elif args.action == "load":
        tokenizer.load(args.pkl_file)

    # Example usage
    #print(tokenizer.encode("floor with one gap. one enemy."))
    #print(tokenizer.get_vocab())
    #for id, token in tokenizer.id_to_token.items():
    #    print(id,":",token)
