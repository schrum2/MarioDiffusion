from interactive_generation import InteractiveGeneration
import torch
from level_dataset import visualize_samples, convert_to_level_format, positive_negative_caption_split
from captions.caption_match import compare_captions, process_scene_segments
from captions.LR_caption_match import compare_captions as lr_compare_captions, process_scene_segments as lr_process_scene_segments
from create_ascii_captions import assign_caption
from LR_create_ascii_captions import assign_caption as lr_assign_caption
from captions.util import extract_tileset
from util.sampler import scene_to_ascii
import argparse
import util.common_settings as common_settings
from util.sampler import SampleOutput
from models.pipeline_loader import get_pipeline
from models.fdm_pipeline import FDMPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate levels using a trained diffusion model")    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--tileset", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--automatic_negative_captions", action="store_true", default=False, help="Automatically create negative captions for prompts so the user doesn't have to")


    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        choices=["Mario", "LR"],
        help="Which game to create a model for (affects sample style and tile count)"
    )

    return parser.parse_args()

class InteractiveLevelGeneration(InteractiveGeneration):
    def __init__(self, args):
        super().__init__(
            {
                "caption": str,
                "width": int,
                "negative_prompt": str,
                "start_seed": int,
                "end_seed": int,
                "num_inference_steps": int,
                "guidance_scale": float
            },
            default_parameters={
                "width":  width, #common_settings.MARIO_WIDTH,
                "start_seed": 1,
                "end_seed": 1,  # Will be set to start_seed if blank
                "num_inference_steps": common_settings.NUM_INFERENCE_STEPS,
                "guidance_scale": common_settings.GUIDANCE_SCALE,
                "caption": "",
                "negative_prompt": ""
            }
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = get_pipeline(args.model_path).to(self.device)
        #self.pipe.print_unet_architecture()
        #self.pipe.save_unet_architecture_pdf(height, width)

        if args.automatic_negative_captions or isinstance(self.pipe, FDMPipeline) or not self.pipe.supports_negative_prompt:
            self.input_parameters.pop('negative_prompt', None)
            self.default_parameters.pop('negative_prompt', None)
        
        if args.automatic_negative_captions and (isinstance(self.pipe, FDMPipeline) or not self.pipe.supports_negative_prompt):
            raise ValueError("Automatic negative caption generation is not possible with a model that doesn't support it")

        if args.tileset:
            _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(args.tileset)

        self.args = args

        if self.args.game == "LR":
            del self.input_parameters["width"]

        print(f"Tileset in use: {self.args.tileset}")

    def generate_image(self, param_values, generator, **extra_params):
        if self.args.automatic_negative_captions:
            pos, neg = positive_negative_caption_split(param_values["caption"], True)
            param_values["negative_prompt"] = neg

        try:
            images = self.pipe(
                generator=generator,
                **param_values
            ).images
        except Exception as e:
            print(f"Error during image generation: {e}")
            return None

        # Convert to indices
        sample_tensor = images[0].unsqueeze(0)
        sample_indices = convert_to_level_format(sample_tensor)

        # Add level data to the list
        scene = sample_indices[0].tolist()
        if self.args.game == "LR":
            number_of_tiles = common_settings.LR_TILE_COUNT
            scene = [[x % number_of_tiles for x in row] for row in scene]
 
        # Assign a caption to the scene of whichever game is being played
        if self.args.game == "Mario":
            actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, self.args.describe_absence)
            level_width = common_settings.MARIO_WIDTH
        elif self.args.game == "LR":
            actual_caption = lr_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, self.args.describe_absence)
            level_width = common_settings.LR_WIDTH
        else:
            raise ValueError(f"Unknown game: {self.args.game}")
        
        if args.game == "LR":
            print(f"Describe resulting image: {actual_caption}")
            lr_compare_score = lr_compare_captions(param_values.get("caption", ""), actual_caption)
            print(f"Comparison score: {lr_compare_score}")

            # Use the new function to process scene segments
            average_score, segment_captions, segment_scores = lr_process_scene_segments(
                scene=scene,
                segment_width=level_width,
                prompt=param_values.get("caption", ""),
                id_to_char=self.id_to_char,
                char_to_id=self.char_to_id,
                tile_descriptors=self.tile_descriptors,
                describe_locations=False, #self.args.describe_locations,
                describe_absence=self.args.describe_absence,
                verbose=True
            )

        elif args.game == "Mario":
            #print(f"Describe resulting image: {actual_caption}")
            compare_score = compare_captions(param_values.get("caption", ""), actual_caption)
            print(f"Comparison score: {compare_score}")

            # Use the new function to process scene segments
            average_score, segment_captions, segment_scores = process_scene_segments(
                scene=scene,
                segment_width=level_width,
                prompt=param_values.get("caption", ""),
                id_to_char=self.id_to_char,
                char_to_id=self.char_to_id,
                tile_descriptors=self.tile_descriptors,
                describe_locations=False, #self.args.describe_locations,
                describe_absence=self.args.describe_absence,
                verbose=True
            )

            # Ask if user wants to play level
            play_level = input("Do you want to play this level? (y/n): ").strip().lower()
            if play_level == 'y':
                print("Playing level...")
                char_grid = scene_to_ascii(scene, self.id_to_char, False)
                level = SampleOutput(level=char_grid, use_snes_graphics=False)
                console_output = level.run_astar()
                print(console_output)
            elif play_level == 'n':
                print("Level not played.")
            else:
                print("Unknown input: Level not played.")
        
        if self.args.game == "Mario":
            samples = visualize_samples(images)
        elif self.args.game == "LR":
            samples = visualize_samples(images, game='LR')

        return samples

    def get_extra_params(self, param_values): 
        if "negative_prompt" in param_values and param_values["negative_prompt"] == "":
            del param_values["negative_prompt"]

        if param_values["caption"] == "":
            del param_values["caption"]

        param_values["output_type"] = "tensor"

        # Lode Runner
        if self.args.game == "LR":
            param_values["height"] = common_settings.LR_HEIGHT
            param_values["width"] = common_settings.LR_WIDTH

        return dict()

if __name__ == "__main__":
    args = parse_args()

    if args.game == "Mario":
        args.num_tiles = common_settings.MARIO_TILE_COUNT
        height = common_settings.MARIO_HEIGHT
        width = common_settings.MARIO_WIDTH
        args.tile_size = common_settings.MARIO_TILE_PIXEL_DIM
        args.tileset = '..\TheVGLC\Super Mario Bros\smb.json'
    elif args.game == "LR":
        args.num_tiles = common_settings.LR_TILE_COUNT
        height = common_settings.LR_HEIGHT
        width = common_settings.LR_WIDTH
        args.tile_size = common_settings.LR_TILE_PIXEL_DIM
        args.tileset = '..\TheVGLC\Lode Runner\Loderunner.json'
    else:
        raise ValueError(f"Unknown game: {args.game}")
    
    ig = InteractiveLevelGeneration(args)
    ig.start()

