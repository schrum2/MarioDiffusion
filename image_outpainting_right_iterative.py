import argparse
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import numpy as np

# Argument parsing
parser = argparse.ArgumentParser(description="Process an image with Stable Diffusion outpainting, gradually expanding rightward.")
parser.add_argument("image", type=str, help="Path to the input image.")
parser.add_argument("-p", "--prompt", type=str, default=None, help="Initial text prompt describing the image.")
parser.add_argument("-n", "--negative_prompt", type=str, 
                    default="", 
                    help="Negative prompt to guide the generation.")
parser.add_argument("-i", "--num_inference_steps", type=int, default=50, 
                    help="Number of inference steps.")
parser.add_argument("-g", "--guidance_scale", type=float, default=8.0, 
                    help="Guidance scale for classifier-free guidance.")
parser.add_argument("-l", "--lora", type=str, default=None, 
                    help="LoRA model to apply.")
parser.add_argument("--seed", type=int, default=1, help="Initial random seed for reproducibility.")

args = parser.parse_args()

# Load the inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

if args.lora:
    pipe.load_lora_weights(
        pretrained_model_name_or_path_or_dict=args.lora,
        adapter_name = "my_lora",
        use_safetensors=True  # Ensures it treats the file as a safetensors file
    )

    pipe.set_adapters(
        ["my_lora"],
        adapter_weights = [1.0]
    )

# Function to infer prompt if not provided
def infer_prompt(image):
    from image_to_text_blip import image_to_text
    generated_text = image_to_text(image, "cuda", True)
    print("BLIP Image Description:", generated_text)
    return generated_text

# Function to draw a boundary line showing new content
def draw_boundary_line(image, boundary_x, line_color=(0, 255, 0), thickness=2):
    """Draw a vertical line on the image to show the boundary between old and new content"""
    img_with_boundary = image.copy()
    draw = ImageDraw.Draw(img_with_boundary)
    draw.line([(boundary_x, 0), (boundary_x, image.height)], fill=line_color, width=thickness)
    return img_with_boundary

# Load initial image
original_image = Image.open(args.image).convert("RGB")
current_prompt = args.prompt
if current_prompt is None:
    current_prompt = infer_prompt(original_image)

# Current working image (will grow with each iteration)
working_image = original_image.copy()
current_seed = args.seed

# Main loop for gradual expansion
iteration = 1
while True:
    print(f"\n--- Expansion Iteration {iteration} ---")
    
    # Get the square portion from the right side of the image
    image_height = working_image.height
    square_size = min(image_height, working_image.width)  
    #print("square_size:",square_size)
    # Extract the rightmost square with overlap
    start_x = max(0, working_image.width - square_size)
    #print("start_x:",start_x)    
    # Store the boundary position for visualization
    boundary_position = working_image.width
    #print("boundary_position",boundary_position)
    generation_accepted = False
    while not generation_accepted:
        # Display current prompt and allow modification
        print(f"Current prompt: {current_prompt}")
        new_prompt = input("Enter new prompt (press Enter to keep current): ")
        if new_prompt.strip():
            current_prompt = new_prompt
        
        # Allow seed modification
        new_seed = input(f"Current seed: {current_seed}. Enter new seed (press Enter to keep current): ")
        if new_seed.strip() and new_seed.isdigit():
            current_seed = int(new_seed)
        
        # Extract the square portion to expand
        square_portion = working_image.crop((start_x, 0, working_image.width, image_height))
        
        # Create a canvas twice as wide as the square
        new_width = square_size * 2
        canvas = Image.new("RGB", (new_width, image_height), (255, 255, 255))
        canvas.paste(square_portion, (0, 0))
        #print("new_width",new_width)
        #canvas.show()

        # Create mask for the right half to be generated
        mask = Image.new("L", (new_width, image_height), 0)  # 0 = Black = keep as is
        mask_draw_area = (square_size, 0, new_width, image_height)
        mask_draw = Image.new("L", (mask_draw_area[2] - mask_draw_area[0], image_height), 255)
        mask.paste(mask_draw, (mask_draw_area[0], mask_draw_area[1]))
        
        #mask.show()

        generator = torch.Generator("cuda").manual_seed(current_seed)
        
        # Run the outpainting pipeline
        print("Generating new segment...")
        expanded_square = pipe(
            prompt=current_prompt,
            negative_prompt=args.negative_prompt,
            image=canvas,
            mask_image=mask,
            height=image_height,
            width=canvas.width,   # Use the new canvas width
            strength=1.0,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        ).images[0]

        #expanded_square.show()
        
        # Create the complete new image by combining the left portion with the newly generated right portion
        exclude_width = start_x 
        left_portion = working_image.crop((0, 0, exclude_width, image_height))
        right_portion = expanded_square

        #print("exclude_width",exclude_width)
        #if exclude_width > 0: left_portion.show()
        #right_portion.show()

        new_total_width = exclude_width + right_portion.width
        complete_image = Image.new("RGB", (new_total_width, image_height), (255, 255, 255))
        complete_image.paste(left_portion, (0, 0))
        complete_image.paste(right_portion, (exclude_width, 0))
        #print("new_total_width",new_total_width)
        # Create version with boundary line
        complete_image_with_boundary = draw_boundary_line(complete_image, boundary_position)
        
        # Show both versions
        print("Showing regular result...")
        complete_image.show()
        print("Showing result with boundary line (green)...")
        complete_image_with_boundary.show()
        
        # Save temporary outputs for both versions
        name, ext = args.image.rsplit('.', 1)
        temp_output = f"{name}_expanded_{iteration}.{ext}"
        temp_output_boundary = f"{name}_expanded_{iteration}_boundary.{ext}"
        complete_image.save(temp_output)
        complete_image_with_boundary.save(temp_output_boundary)
        print(f"Temporary results saved as {temp_output} and {temp_output_boundary}")
        
        # Ask if user wants to keep the generation or try again
        choice = input("Keep this generated segment? (y/n): ").lower()
        if choice == 'y':
            generation_accepted = True
        else:
            print("Let's try generating a different segment.")
            # Seed will be changed automatically if not explicitly changed by user
            current_seed += 1
    
    # Ask if user wants to continue adding more segments
    continue_choice = input("Continue adding more segments? (y/n): ").lower()
    if continue_choice != 'y':
        # Save final outputs
        final_output = f"{name}_FINAL.{ext}"
        final_output_boundary = f"{name}_FINAL_boundary.{ext}"
        complete_image.save(final_output)
        complete_image_with_boundary.save(final_output_boundary)
        print(f"Final images saved as {final_output} and {final_output_boundary}")
        break
    
    # Update working image for next iteration
    working_image = complete_image.copy()
    current_seed += 1  # Increment seed for variety in next segment
    iteration += 1

print("Outpainting process completed.")