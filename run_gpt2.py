import torch
import os
from mario_gpt import MarioLM, SampleOutput

# Path to your trained model and tokenizer
MODEL_DIR = "Mario-GPT2-TEST-MODEL"
CHECKPOINT = os.path.join(MODEL_DIR, "model_200000.pt")  # Use the latest checkpoint

# Initialize MarioLM and load tokenizer if available
mario_lm = MarioLM()
if hasattr(mario_lm, "tokenizer") and hasattr(mario_lm.tokenizer, "from_pretrained"):
    mario_lm.tokenizer = mario_lm.tokenizer.from_pretrained(MODEL_DIR)

# Load model weights (assumes MarioLM.lm is the HuggingFace model)
state_dict = torch.load(CHECKPOINT, map_location="cpu")
print("Before loading:", mario_lm.lm.transformer.h[0].attn.c_attn.weight[0][:5])
mario_lm.lm.load_state_dict(state_dict)
print("After loading:", mario_lm.lm.transformer.h[0].attn.c_attn.weight[0][:5])
print("Loaded custom trained weights from", CHECKPOINT)

# Optionally move to CUDA if available
if torch.cuda.is_available():
    mario_lm.lm = mario_lm.lm.to("cuda")

prompts = ["many pipes, many enemies, some blocks, high elevation"]

# Generate a level
generated_level = mario_lm.sample(
    prompts=prompts,
    num_steps=1400,
    temperature=2.0,
    use_tqdm=True
)

# Show string list
print(generated_level.level)

# Show PIL image
generated_level.img.show()

# Save image
generated_level.img.save("generated_level.png")

# Save text level to file
generated_level.save("generated_level.txt")

# Play in interactive
generated_level.play()

# Run Astar agent
generated_level.run_astar()

# Continue generation
generated_level_continued = mario_lm.sample(
    seed=generated_level,
    prompts=prompts,
    num_steps=1400,
    temperature=2.0,
    use_tqdm=True
)

# Load from text file
loaded_level = SampleOutput.load("generated_level.txt")

# Play from loaded (should be the same level that we generated)
loaded_level.play()