from evolution.evolution import Evolver
from level_dataset import visualize_samples, convert_to_level_format
from create_ascii_captions import extract_tileset
import argparse
import torch
from evolution.genome import LatentGenome
from create_ascii_captions import assign_caption
import util.common_settings as common_settings
from models.pipeline_loader import get_pipeline


class TextDiffusionEvolver(Evolver):
    def __init__(self, model_path, width, tileset_path='datasets\smb.json', args = None):
        Evolver.__init__(self)
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

    def random_latent(self, seed=1):
        # Create the initial noise latents (this is what the pipeline does internally)
        if args.tileset == '..\TheVGLC\Super Mario Bros\smb.json':
            height = common_settings.MARIO_HEIGHT
            width = self.width
        elif args.tileset == '..\TheVGLC\Lode Runner\Loderunner.json':
            height = common_settings.LR_HEIGHT
            width = common_settings.LR_WIDTH
        num_channels_latents = len(self.id_to_char)
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
        generator = torch.Generator("cuda").manual_seed(g.seed)
        settings = {
            "guidance_scale": g.guidance_scale, 
            "num_inference_steps": g.num_inference_steps,
            "output_type": "tensor",
            "raw_latent_sample": g.latents.to("cuda")
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

        actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, self.args.describe_absence)
        g.caption = actual_caption

        if args.tileset == '..\TheVGLC\Super Mario Bros\smb.json':
            samples = visualize_samples(images)
        elif args.tileset == '..\TheVGLC\Lode Runner\Loderunner.json':
            samples = visualize_samples(images, game='LR')
        return samples

def parse_args():
    parser = argparse.ArgumentParser(description="Evolve levels with unconditional diffusion model")    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--tileset_path", default='datasets\smb.json', help="Descriptions of individual tile types")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--width", type=int, default=common_settings.MARIO_WIDTH, help="Tile width of generated level")

    return parser.parse_args()

if __name__ == "__main__": 
    args = parse_args()
    evolver = TextDiffusionEvolver(args.model_path, args.width, args.tileset_path, args=args)
    allow_negative_prompt = getattr(evolver.pipe, "supports_negative_prompt", False)
    evolver.start_evolution(allow_prompt=True, allow_negative_prompt=allow_negative_prompt)