import torch
from mario_gpt import MarioDataset, MarioLM
from MarioGPT_train_helper import TrainingConfig, MarioGPTTrainer  # <-- import from helper file

print("Initializing MarioGPT model...")
# create basic gpt model
BASE = "distilgpt2"
mario_lm = MarioLM()
print("Model initialized.")

print("Loading Mario level dataset...")
# create dataset
dataset = MarioDataset(mario_lm.tokenizer)
print(f"Dataset loaded with {len(dataset)} samples.")

print("Setting up training configuration and trainer...")
# create training config and trainer
config = TrainingConfig(save_iteration=10)
trainer = MarioGPTTrainer(mario_lm, dataset, config=config)
print("Trainer ready.")

print("Starting training for 100 iterations...")
# train for 100 iterations!
trainer.train(100, batch_size=1)
print("Training complete!")


