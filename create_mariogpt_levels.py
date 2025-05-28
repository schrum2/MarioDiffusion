from mario_gpt import MarioLM
import os
import argparse

# pretrained_model = shyamsn97/Mario-GPT2-700-context-length
mario_lm = MarioLM()

PROMPTS = ["floor with one gap, one question block, two cannons, two platforms, one tower, one irregular block cluster"]

def generate_levels(total_samples=10, width=100, temperature=2.0):
    prompts_list = PROMPTS * total_samples  # Repeat the prompt to generate multiple levels
    num_steps = width * 14
    generated_levels = mario_lm.sample(
        prompts=prompts_list,
        num_steps=num_steps,
        temperature=temperature,
        use_tqdm=True
    )
    return generated_levels

def save_data(generated_levels, output_dir):
    # Save levels to the output directory
    for i, level in enumerate(generated_levels):
        img_path = os.path.join(output_dir, f"level_{i+1}.png")
        txt_path = os.path.join(output_dir, f"level_{i+1}.txt")

        level.img.save(img_path)
        with open(txt_path, "w") as f:
            f.write("\n".join(level.level))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mario levels using MarioGPT.")
    parser.add_argument("--samples", type=int, default=10, help="Number of levels to generate")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "MarioGPT-samples"), help="Directory to save generated levels")
    parser.add_argument("--width", type=int, default=100, help="Width of the level (actual num_steps = width * 14)")
    parser.add_argument("--temp", type=float, default=2.0, help="Sampling temperature for level generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generated_levels = generate_levels(total_samples=args.samples, width=args.width, temperature=args.temp)
    save_data(generated_levels, args.output_dir)
    print(f"Levels saved to {args.output_dir}")
