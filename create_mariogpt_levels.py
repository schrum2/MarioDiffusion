from mario_gpt import MarioLM, SampleOutput
import os

# pretrained_model = shyamsn97/Mario-GPT2-700-context-length
mario_lm = MarioLM()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "MarioGPT-samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# This should generate 15 levels, each with a different prompt
prompts = ["floor with one gap, one question block, two cannons, two platforms, one tower, one irregular block cluster"]

def generate_levels(n=10):
    # generate level of size 1400, pump temperature up to ~2.4 for more stochastic but playable levels
    generated_levels = mario_lm.sample(
        prompts=prompts,
        num_steps=1400,
        temperature=2.0,
        use_tqdm=True
    )
    return generated_levels

def save_data(generated_levels):
    # Save levels to the output directory
    for i, level in enumerate(generated_levels):
        img_path = os.path.join(OUTPUT_DIR, f"level_{i+1}.png")
        txt_path = os.path.join(OUTPUT_DIR, f"level_{i+1}.txt")

        level.img.save(img_path)
        with open(txt_path, "w") as f:
            f.write("\n".join(level.level))

if __name__ == "__main__":
    generated_levels = generate_levels()
    save_data(generated_levels)
    print(f"Levels saved to {OUTPUT_DIR}")
