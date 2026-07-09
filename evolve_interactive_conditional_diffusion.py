from evolution.evolution import Evolver
from level_dataset import visualize_samples, convert_to_level_format
from create_ascii_captions import extract_tileset
import argparse
import torch
from evolution.genome import LatentGenome
from create_ascii_captions import assign_caption
from LR_create_ascii_captions import assign_caption as lr_assign_caption
from captions.caption_match import compare_captions
from captions.MM2_caption_match import caption_tools as mm2_caption_tools
import util.common_settings as common_settings
from models.pipeline_loader import get_pipeline


class TextDiffusionEvolver(Evolver):
    def __init__(self, model_path, width, tileset_path=common_settings.MARIO_TILESET, args = None):
        Evolver.__init__(self, args)
        # args = parse_args()
        # if args.tileset_path != "":
        #     tileset_path = args.tileset_path
        #print("tile path:", tileset_path)
        self.args = args
        self.width = width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = get_pipeline(model_path).to(self.device)
        # Set negative prompt support in viewer if available
        self.negative_prompt_supported = getattr(self.pipe, "supports_negative_prompt", False)

        _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)
        # print("self.id_to_char:", self.id_to_char)
        # print("self.char_to_id:", self.char_to_id)

        # MM2 gets its own captioner + comparison (read from the tileset, not the
        # Mario tag table), matching run_diffusion / evaluate_caption_adherence.
        if args is not None and args.game == 'MM2':
            self.mm_assign_caption, self.mm_compare_captions = mm2_caption_tools(tileset_path)

    def random_latent(self, seed=1):
        # Create the initial noise latents (this is what the pipeline does internally)
        if self.args.game == 'Mario':
            height = common_settings.MARIO_HEIGHT
            width = self.width
        elif self.args.game == 'MM2':
            height = common_settings.MM2_HEIGHT
            width = self.width
        elif self.args.game == 'LR':
            height = common_settings.LR_HEIGHT
            width = common_settings.LR_WIDTH
        num_channels_latents = self.pipe.unet.config.in_channels
        #print("num_channels_latents:", num_channels_latents)
        latents_shape = (1, num_channels_latents, height, width)
        latents = torch.randn(
            latents_shape, 
            generator=torch.manual_seed(seed)        
        ).to("cpu")
        return latents

    def initialize_population(self):
        self.genomes = [LatentGenome(self.width, seed, self.steps, self.guidance_scale, latents=self.random_latent(seed), prompt=self.prompt, negative_prompt=self.negative_prompt, num_segments=1) for seed in range(self.population_size)]
        self.viewer.id_to_char = self.id_to_char

    def generate_image(self, g):
        # generate fresh new image
        print(f"Generate new image for {g}")
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(g.seed)
        settings = {
            "guidance_scale": g.guidance_scale, 
            "num_inference_steps": g.num_inference_steps,
            "output_type": "tensor",
            "raw_latent_sample": g.latents.to("cuda" if torch.cuda.is_available() else "cpu")
        }
        # Include caption if desired
        if g.prompt and g.prompt.strip() != "":
            settings["caption"] = g.prompt

        # Include negative prompt if supported and provided
        if getattr(self.pipe, "supports_negative_prompt", False):
            neg_prompt = g.negative_prompt
            if neg_prompt is not None and neg_prompt.strip() != "":
                settings["negative_prompt"] = neg_prompt

        images = self.pipe(
            generator=generator,
            **settings
        ).images
        g.latents.to("cpu")
        # Convert to indices
        sample_indices = convert_to_level_format(images)
        
        # Add level data to the list
        scene = sample_indices[0].tolist() # Always just one scene: (1,16,16)
        g.scene = scene 
        if args.game == 'Mario':
            actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, self.args.describe_absence)
        elif args.game == 'MM2':
            actual_caption = self.mm_assign_caption(scene)
        elif args.game == 'LR':
            actual_caption = lr_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, self.args.describe_absence)
        g.caption = actual_caption

        # Score how well the generated scene matches the prompt it was conditioned
        # on: compare the prompt against the scene's own deterministic caption.
        if g.prompt and g.prompt.strip():
            if args.game == 'MM2':
                g.score = self.mm_compare_captions(g.prompt, actual_caption)
            elif args.game == 'Mario':
                g.score = compare_captions(g.prompt, actual_caption)
            else:
                g.score = None
        else:
            g.score = None
        print(f"Caption adherence score: {g.score}")

        if args.game == 'Mario':
            samples = visualize_samples(images)
        elif args.game == 'MM2':
            # game='MM2' renders the real MM2 sprites instead of flat Mario tiles
            samples = visualize_samples(images, game='MM2')
        elif args.game == 'LR':
            samples = visualize_samples(images, game='LR')
        return samples

def parse_args():
    parser = argparse.ArgumentParser(description="Evolve levels with unconditional diffusion model")    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--tileset_path", default=common_settings.MARIO_TILESET, help="Descriptions of individual tile types")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--width", type=int, default=common_settings.MARIO_WIDTH, help="Tile width of generated level")

    parser.add_argument(
        "--game",
        type=str,
        default="MM2",
        choices=["Mario", "MM2", "LR", "MM-Simple", "MM-Full"],
        help="Which game to create a model for (affects sample style and tile count)"
    )

    return parser.parse_args()

if __name__ == "__main__": 
    args = parse_args()

    if args.game == "Mario":
        args.tileset_path = common_settings.MARIO_TILESET
    elif args.game == 'MM2':
        # Mario Maker 2: mm2_tileset_we.json, 20x20 scenes
        args.tileset_path = common_settings.MM2_TILESET
        args.width = common_settings.MM2_WIDTH
    elif args.game == 'LR':
        args.tileset_path = common_settings.LR_TILESET
        args.width = common_settings.LR_WIDTH
    elif args.game == 'MM-Simple':
        args.tileset_path = 'datasets/MM-simple-tileset.json'
    elif args.game == 'MM-Full':
        args.tileset_path = '../TheVGLC/MegaMan/MM.json'

    evolver = TextDiffusionEvolver(args.model_path, args.width, args.tileset_path, args=args)
    allow_negative_prompt = getattr(evolver.pipe, "supports_negative_prompt", False)
    evolver.start_evolution(allow_prompt=True, allow_negative_prompt=allow_negative_prompt)