import os
import sys
from PIL import Image

# Created by ChatGPT

def process_mario_levels(input_dir, output_dir, step_size):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            level_img = Image.open(input_path).convert("RGB")

            width, height = level_img.size
            background_color = level_img.getpixel((0, 0))  # Assume top-left pixel is the background

            # Expand height to 256 by adding background color padding
            expanded_img = Image.new("RGB", (width, 256), background_color)
            #expanded_img.paste(level_img, (0, (256 - height) // 2))
            # Paste to bottom, not middle
            expanded_img.paste(level_img, (0, 256 - height))


            # Slide 256×256 window across the width at step_size pixel intervals
            counter = 0
            for x in range(0, width - 256 + 1, step_size):
                cropped_img = expanded_img.crop((x, 0, x + 256, 256))
                output_filename = f"{os.path.splitext(filename)[0]}_{counter:04d}.png"
                cropped_img.save(os.path.join(output_dir, output_filename))
                counter += 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_mario_levels.py <input_dir> <output_dir> [<window step = 16>]")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    step_size = 16 if len(sys.argv) == 3 else int(sys.argv[3])

    process_mario_levels(input_directory, output_directory, step_size)
