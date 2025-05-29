import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from mario_gpt import MarioDataset, MarioLM
from train_gpt2_helper import TrainingConfig, MarioGPTTrainer

print("Initializing MarioGPT model...")
BASE = "distilgpt2"

# Modify config to enable cross-attention
gpt_config = GPT2Config.from_pretrained(BASE)
gpt_config.add_cross_attention = True

tokenizer = AutoTokenizer.from_pretrained(BASE)

model = GPT2LMHeadModel.from_pretrained(BASE, config=gpt_config)

mario_lm = MarioLM(lm=model, tokenizer=tokenizer)
print("Model initialized.")

print("Loading Mario level dataset...")
# create dataset
dataset = MarioDataset(mario_lm.tokenizer)
print(f"Dataset loaded with {len(dataset)} samples.")

print("Setting up training configuration and trainer...")
# create training config and trainer
train_config = TrainingConfig(save_iteration=10)
trainer = MarioGPTTrainer(mario_lm, dataset, config=train_config)
print("Trainer ready.")

# # Inference before training
# print("Generating a sample level before training...")
# prompts = ["many pipes, many enemies, some blocks, high elevation"]

# # generate level of size 1400, pump temperature up to ~2.4 for more stochastic but playable levels
# generated_level = mario_lm.sample(
#     prompts=prompts,
#     num_steps=1400,
#     temperature=2.0,
#     use_tqdm=True
# )

# # Show sample
# print("Generated level (string representation):")
# print(generated_level.level)
# print("Generated level (PIL image):")
# generated_level.img.show()
# print("Pretraining sample level generation complete.")

print("Starting training for 100 iterations...")
# train for 100 iterations!
trainer.train(50000, batch_size=1)
print("Training complete!")