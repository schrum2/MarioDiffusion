import json
import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer

class LevelDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=50):
        """
        Dataset class for handling level data with dynamic tokenization.
        
        Args:
            json_path (str): Path to the JSON file containing level scenes and captions.
            tokenizer (Tokenizer): Tokenizer instance for processing captions.
            max_length (int): Maximum length for tokenized sequences (truncation/padding).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load JSON data (list of dicts)
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

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
        caption_tokens = caption_tokens[:self.max_length]  # Truncate if too long
        caption_tokens += [0] * (self.max_length - len(caption_tokens))  # Pad if too short
        
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

    dataset = LevelDataset('SMB1_LevelsAndCaptions.json', tokenizer)

    # Fetch a sample
    scene_tensor, caption_tensor = dataset[0]

    # Print raw and tokenized output
    print("Raw Caption:", dataset.get_sample_caption(0))
    print("Tokenized Caption:", caption_tensor.tolist())
    print("Decoded Caption:", dataset.decode_caption(caption_tensor.tolist()))
    print("Vocab Size:", dataset.get_vocab_size())

    # Find the longest caption
    longest_idx = max(range(len(dataset)), key=lambda i: len(dataset.get_sample_caption(i)))

    # Fetch the longest caption
    longest_caption = dataset.get_sample_caption(longest_idx)
    tokenized_caption = dataset.tokenizer.encode(longest_caption)

    # Print results
    print("Index of Longest Caption:", longest_idx)
    print("Raw Caption:", longest_caption)
    print("Tokenized Length:", len(tokenized_caption))
    print("Tokenized Caption:", tokenized_caption)
    print("Decoded Caption:", dataset.decode_caption(tokenized_caption))