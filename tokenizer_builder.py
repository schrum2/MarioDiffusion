import json
import os
from collections import Counter
from typing import List, Dict, Optional

class TokenizerBuilder:
    def __init__(self):
        self.vocab_counter = Counter()
        self.vocab = []
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize_caption(self, caption: str) -> List[str]:
        """
        Simple whitespace tokenizer. Replace or extend with regex if needed.
        """
        return caption.lower().strip().replace('.', '').replace(',', '').split()

    def scan_dataset(self, json_path: str):
        """
        Scans the dataset JSON and builds the vocabulary counter.
        Expects a list of dicts with 'caption' fields.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            caption = entry.get('caption', '')
            tokens = self.tokenize_caption(caption)
            self.vocab_counter.update(tokens)

    def finalize_vocab(self, special_tokens: Optional[List[str]] = None):
        """
        Finalizes the vocabulary list, adding special tokens if provided.
        Creates token-to-id and id-to-token mappings.
        """
        self.vocab = []
        if special_tokens:
            self.vocab.extend(special_tokens)

        sorted_tokens = [token for token, _ in self.vocab_counter.most_common()]
        self.vocab.extend(sorted_tokens)

        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def save_vocab(self, output_path: str):
        """
        Saves the vocabulary as a JSON file for later use.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2)

    def load_vocab(self, vocab_path: str):
        """
        Loads the vocabulary from a JSON file.
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, caption: str) -> List[int]:
        """
        Converts a caption to a list of token IDs.
        """
        tokens = self.tokenize_caption(caption)
        return [self.token_to_id.get(token, self.token_to_id.get('<unk>', -1)) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        Converts a list of token IDs back to a caption string.
        """
        tokens = [self.id_to_token.get(idx, '<unk>') for idx in token_ids]
        return ' '.join(tokens)

