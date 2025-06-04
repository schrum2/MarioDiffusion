import torch
from mario_gpt import MarioDataset, MarioLM
from util.trainer import TrainingConfig, MarioGPTTrainer
from transformers import AutoModelWithLMHead, AutoTokenizer



# create basic gpt model
BASE = "distilgpt2"
lm = AutoModelWithLMHead.from_pretrained(BASE, add_cross_attention = True)

tokenizer = AutoTokenizer.from_pretrained(BASE)

mario_lm = MarioLM(lm=lm, tokenizer=tokenizer)

# create dataset
dataset = MarioDataset(mario_lm.tokenizer)

# create training config and trainer
config = TrainingConfig()
trainer = MarioGPTTrainer(mario_lm, dataset, config=config)

# train for 100 iterations!
trainer.train(50000, batch_size=1)
