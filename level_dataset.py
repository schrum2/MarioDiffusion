import json
import torch
import random
from torch.utils.data import Dataset
from tokenizer import Tokenizer


class LevelDataset:
    def __init__(self, json_path, tokenizer, batch_size=32, shuffle=True, max_length=None, mode="diffusion"):
        """
        Args:
            json_path (str): Path to JSON file with captions.
            tokenizer (Tokenizer): Tokenizer instance.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data at the start of an epoch.
            max_length (int, optional): Maximum length for tokenized captions.
            mode (str): "diffusion" for level scenes + captions, "mlm" for masked language model training.
        """
        assert mode in ["mlm", "diffusion"], "Mode must be 'mlm' or 'diffusion'."
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

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
        pad_token = self.tokenizer.token_to_id["[PAD]"]
        return tokens + [pad_token] * (self.max_length - len(tokens))

    def get_batch(self, idx):
        """
        Retrieves a batch of data.

        Args:
            idx (int): Batch index.

        Returns:
            - In "mlm" mode: token_tensor (Tensor) of shape (batch_size, max_length)
            - In "diffusion" mode: (scenes_tensor, captions_tensor)
        """
        start = idx * self.batch_size
        end = start + self.batch_size

        batch_tokens = self.tokenized_captions[start:end]
        batch_tokens = [self._pad_sequence(tokens) for tokens in batch_tokens]
        caption_tensor = torch.tensor(batch_tokens, dtype=torch.long)

        if self.mode == "mlm":
            return caption_tensor  # MLM only uses captions

        # Get level scenes for diffusion training
        batch_scenes = [self.data[i]["scene"] for i in range(start, min(end, len(self.data)))]
        scene_tensor = torch.tensor(batch_scenes, dtype=torch.long)

        return scene_tensor, caption_tensor

    def __len__(self):
        """Returns number of batches."""
        return len(self.tokenized_captions) // self.batch_size

    def __getitem__(self, idx):
        """
        Fetches one sample.

        Returns:
            - In "mlm" mode: tokenized caption
            - In "diffusion" mode: (scene_tensor, caption_tensor)
        """
        sample = self.data[idx]
        caption_tokens = self.tokenizer.encode(sample["caption"])
        caption_tokens = self._pad_sequence(caption_tokens)
        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)

        if self.mode == "mlm":
            return caption_tensor  # MLM only uses captions

        scene_tensor = torch.tensor(sample["scene"], dtype=torch.long)  # Convert scene to tensor
        return scene_tensor, caption_tensor

    def decode_caption(self, token_ids):
        """Converts a sequence of token IDs back into a readable caption."""
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self):
        """Returns the size of the tokenizer vocabulary."""
        return len(self.tokenizer.get_vocab())

    def get_sample_caption(self, idx):
        """Returns the raw caption from the dataset for debugging."""
        return self.data[idx]["caption"]


if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.load('SMB1_Tokenizer.pkl')

    # Create MLM dataset
    mlm_dataset = LevelDataset('SMB1_LevelsAndCaptions.json', tokenizer, batch_size=16, mode="mlm")
    batch = mlm_dataset.get_batch(0)
    print("MLM Batch Shape:", batch.shape)  # Should be (16, max_length)

    # Create Diffusion dataset
    diffusion_dataset = LevelDataset('SMB1_LevelsAndCaptions.json', tokenizer, batch_size=16, mode="diffusion")
    scenes, captions = diffusion_dataset.get_batch(0)
    print("Diffusion Batch Shapes:", scenes.shape, captions.shape)  # Expected: (16,16,16) and (16, max_length)
