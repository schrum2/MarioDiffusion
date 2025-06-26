import json
import argparse
import numpy as np
from typing import List, Dict
import torch
from PIL import Image
from tqdm import tqdm
from models.pipeline_loader import get_pipeline



def load_scenes(json_path: str) -> List[Dict]:
    with open(json_path, 'r') as f:
        return json.load(f)

def mask_right_half(scene: List[List[int]], mask_token: int = 15) -> Dict:
    scene_np = np.array(scene, dtype=np.int64)
    input_scene = scene_np.copy()
    mask = np.ones_like(scene_np, dtype=np.uint8)

    # Mask right 8 columns (columns 8 to 15 inclusive)
    input_scene[:, 8:] = mask_token
    mask[:, 8:] = 0  # Mask = 0 means masked/outpaint area

    return {
        "input": input_scene.tolist(),
        "mask": mask.tolist(),
        "target": scene,
    }

def create_outpainting_dataset(entries: List[Dict], mask_token: int = -1) -> List[Dict]:
    dataset = []
    for entry in entries:
        result = mask_right_half(entry["scene"], mask_token)
        result["caption"] = entry.get("caption", "")
        dataset.append(result)
    return dataset

def save_dataset(dataset: List[Dict], output_path: str):
    with open(output_path, 'w') as f:
        json.dump(dataset, f)

def tensor_to_image(arr: np.ndarray, scale: int = 16) -> Image.Image:
    """Upscale 2D array into a grayscale PIL image"""
    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    return img.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)

def generate_right_half_samples(
    dataset: List[Dict],
    output_path: str,
    model_path: str,
    device: str = "cuda"
):
    


    # Load trained diffusion pipeline
    pipe = get_pipeline(model_path).to(device)
    pipe.set_progress_bar_config(disable=True)

    generated_samples = []

    for entry in tqdm(dataset):
        # Convert inputs to proper dtypes
        input_np = np.array(entry["input"], dtype=np.int64)    # shape (16,16)
        mask_np = np.array(entry["mask"], dtype=np.uint8)      # shape (16,16), 1=known, 0=masked
        caption = entry.get("caption", "")

        # Convert to torch tensors, add batch dim, move to device
        input_tensor = torch.tensor(input_np, dtype=torch.long, device=device).unsqueeze(0)
        mask_tensor = torch.tensor(mask_np, dtype=torch.bool, device=device).unsqueeze(0)

        # Run diffusion pipeline conditioned on input scene and mask
        output = pipe(
            caption=caption,
            input_scene=input_tensor,
            mask=mask_tensor,
            generator=torch.manual_seed(42),
            output_type="tensor",
            show_progress_bar=False,
        )

        # output shape expected: [batch, height, width] or [batch, channels, H, W]
        # Remove batch dimension, convert to CPU numpy array
        if isinstance(output, torch.Tensor):
            output_arr = output.squeeze(0).cpu().numpy()
        else:
            # If output is PIL Image or list, convert accordingly (depends on pipeline impl)
            output_arr = np.array(output)

        # Clip or convert output to int64 (tile IDs)
        output_arr = output_arr.astype(np.int64)
        
        # Reconstruct full scene: use input for left, generated for right
        reconstructed_scene = input_np.copy()
        reconstructed_scene[:, 8:] = output_arr[:, 8:]

        generated_samples.append({
            "input": input_np.tolist(),
            "mask": mask_np.tolist(),
            "caption": caption,
            "generated_half": output_arr.tolist(),   # just the generated region
            "reconstructed_scene": reconstructed_scene.tolist()
        })

    # Save generated samples
    with open(output_path, 'w') as f:
        json.dump(generated_samples, f)

    print(f"Saved generated results to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Outpainting Mario levels using diffusion model.")
    parser.add_argument('--input_json', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--model_path', type=str, help="Path to TextConditionalDDPMPipeline model")
    parser.add_argument('--generate', action='store_true', help="Whether to generate outputs")
    parser.add_argument('--device', type=str, default="cuda", help="Device to run model on, e.g., cuda or cpu")
    args = parser.parse_args()

    entries = load_scenes(args.input_json)
    dataset = create_outpainting_dataset(entries)

    generate_right_half_samples(dataset, args.output_json, args.model_path, args.device)

if __name__ == "__main__":
    main()