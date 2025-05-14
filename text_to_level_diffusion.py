from interactive_generation import InteractiveGeneration
import torch
from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples, convert_to_level_format
from captions.caption_match import compare_captions, process_scene_segments
from create_ascii_captions import assign_caption, extract_tileset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate levels using a trained diffusion model")    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--tileset", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")

    return parser.parse_args()

class InteractiveLevelGeneration(InteractiveGeneration):
    def __init__(self, args):
        InteractiveGeneration.__init__(self, {
            "caption" : str,
            "width" : int,
            "negative_prompt" : str,
            "start_seed" : int,
            "end_seed" : int,
            "num_inference_steps" : int,
            "guidance_scale" : float
        })

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to(self.device)
        self.pipe.print_unet_architecture()

        if args.tileset:
            _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(args.tileset)

        self.args = args

    def generate_image(self, param_values, generator, **extra_params):
        images = self.pipe(
            generator=generator,
            **param_values
        ).images

        # Convert to indices
        sample_tensor = images[0].unsqueeze(0)
        sample_indices = convert_to_level_format(sample_tensor)
        
        # Add level data to the list
        scene = sample_indices[0].tolist()
 
        actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, self.args.describe_locations, self.args.describe_absence)

        print(f"Describe resulting image: {actual_caption}")
        compare_score = compare_captions(param_values["caption"], actual_caption)
        print(f"Comparison score: {compare_score}")

        # Use the new function to process scene segments
        average_score, segment_captions, segment_scores = process_scene_segments(
            scene=scene,
            segment_width=16,
            prompt=param_values["caption"],
            id_to_char=self.id_to_char,
            char_to_id=self.char_to_id,
            tile_descriptors=self.tile_descriptors,
            describe_locations=self.args.describe_locations,
            describe_absence=self.args.describe_absence,
            verbose=True
        )

        return visualize_samples(images)

    def get_extra_params(self, param_values): 
        if param_values["negative_prompt"] == "":
            del param_values["negative_prompt"]

        if param_values["caption"] == "":
            del param_values["caption"]

        param_values["output_type"] = "tensor"
        return dict()

if __name__ == "__main__":
    args = parse_args()
    ig = InteractiveLevelGeneration(args)
    ig.start()

