import json
import torch
import random
from torch.utils.data import Dataset
from tokenizer import Tokenizer


class LevelDataset:
    def __init__(self, json_path, tokenizer, batch_size=32, shuffle=True, max_length=None):
        """
        Args:
            json_path (str): Path to JSON file with captions.
            tokenizer (Tokenizer): Tokenizer instance.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data at the start of an epoch.
            max_length (int, optional): Maximum length for tokenized captions.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Tokenize all captions in advance
        self.tokenized_captions = [self.tokenizer.encode(entry["caption"]) for entry in self.data]

        # Ensure all tokenized captions are lists of integers
        for i, tokens in enumerate(self.tokenized_captions):
            if not all(isinstance(token, int) for token in tokens):
                raise ValueError(f"Tokenization error at index {i}: {tokens}")

        # Determine padding length (if not provided)
        if self.max_length is None:
            self.max_length = max(len(tokens) for tokens in self.tokenized_captions)

        # Shuffle dataset
        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        """Shuffles the dataset."""
        combined = list(zip(self.data, self.tokenized_captions))
        random.shuffle(combined)
        self.data, self.tokenized_captions = zip(*combined)

    def _pad_sequence(self, tokens):
        """Pads tokenized sequences to max_length with a padding token (assumed to be '[PAD]')."""
        #print(self.tokenizer.token_to_id)
        pad_token = self.tokenizer.token_to_id["[PAD]"]
        return tokens + [pad_token] * (self.max_length - len(tokens))

    def get_batch(self, idx):
        """
        Retrieves a batch of data.
        
        Args:
            idx (int): Batch index.

        Returns:
            token_tensor (Tensor): Tensor of shape (batch_size, max_length).
        """
        start = idx * self.batch_size
        end = start + self.batch_size

        batch_tokens = self.tokenized_captions[start:end]
        batch_tokens = [self._pad_sequence(tokens) for tokens in batch_tokens]

        return torch.tensor(batch_tokens, dtype=torch.long)

    def __len__(self):
        """Returns number of batches."""
        return len(self.tokenized_captions) // self.batch_size

    def __getitem__(self, idx):
        """
        Fetches one sample, tokenizing the caption dynamically.
        
        Returns:
            - scene_tensor: Tensor of shape (16,16) representing the level scene.
            - caption_tensor: Tensor of tokenized caption.
        """
        sample = self.data[idx]
        scene = torch.tensor(sample["scene"], dtype=torch.long)  # Convert to tensor
        caption = sample["caption"]
        
        # Tokenize caption dynamically
        caption_tokens = self.tokenizer.encode(caption)
        
        # Ensure fixed-length output
        pad_token = self.tokenizer.token_to_id["[PAD]"]
        caption_tokens = caption_tokens[:self.max_length]  # Truncate if too long
        caption_tokens += [pad_token] * (self.max_length - len(caption_tokens))
        
        return scene, torch.tensor(caption_tokens, dtype=torch.long)

    def decode_caption(self, token_ids):
        """
        Converts a sequence of token IDs back into a readable caption.
        
        Args:
            token_ids (list[int]): List of token IDs.
        
        Returns:
            str: Decoded caption.
        """
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self):
        """
        Returns the size of the tokenizer vocabulary.
        """
        return len(self.tokenizer.get_vocab())

    def get_sample_caption(self, idx):
        """
        Returns the raw caption from the dataset for debugging.
        
        Args:
            idx (int): Sample index.
        
        Returns:
            str: Original caption text.
        """
        return self.data[idx]["caption"]

if __name__ == "__main__":
    tokenizer = Tokenizer()
    #tokenizer.build_vocab('SMB1_LevelsAndCaptions.json')
    tokenizer.load('SMB1_Tokenizer.pkl')

    dataset = LevelDataset('SMB1_LevelsAndCaptions.json', tokenizer, batch_size=16)

    # Retrieve a batch
    batch_idx = 0
    batch = dataset.get_batch(batch_idx)
    print(batch.shape)  # Expected: (16, max_length)

    print(batch)

    print(dataset[0])