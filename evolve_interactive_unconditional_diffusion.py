from evolution.evolution import Evolver
from level_dataset import visualize_samples, convert_to_level_format
from create_ascii_captions import extract_tileset
import argparse
import torch
from evolution.genome import LatentGenome
from create_ascii_captions import assign_caption
from LR_create_ascii_captions import assign_caption as lr_assign_caption
from captions.MM2_caption_match import caption_tools as mm2_caption_tools
import util.common_settings as common_settings
from models.pipeline_loader import get_pipeline


class DiffusionEvolver(Evolver):
    def __init__(self, model_path, width, tileset_path=common_settings.MARIO_TILESET, args=None):
        Evolver.__init__(self, args)

        self.args = args
        self.width = width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = get_pipeline(model_path).to(self.device)

        #self.pipe.print_unet_architecture()
        _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)

        # MM2 captions read tile names from the tileset, not the Mario tag table
        if args is not None and args.game == 'MM2':
            self.mm_assign_caption, _ = mm2_caption_tools(tileset_path)

    def random_latent(self, seed=1):

        config = common_settings.get_game_config(self.args.game)
        height = config["height"]
        width = config["width"]
        num_channels_latents = config["tile_count"]

        # Create the initial noise latents (this is what the pipeline does internally)
        latents_shape = (1, num_channels_latents, height, width)
        latents = torch.randn(
            latents_shape, 
            generator=torch.manual_seed(seed)        
        ).to("cpu")
        return latents

    def initialize_population(self):
        self.genomes = [LatentGenome(self.width, seed, self.steps, self.guidance_scale, latents=self.random_latent(seed), num_segments=1) for seed in range(self.population_size)]
        # Removed generation_width from LatentGenome constructor
        self.viewer.id_to_char = self.id_to_char

    def generate_image(self, g):
        # generate fresh new image
        print(f"Generate new image for {g}")
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(g.seed)

        settings = {
            "batch_size" : 1,
            # "guidance_scale" : g.guidance_scale, # Remove this from genome?
            "num_inference_steps" : g.num_inference_steps,
            # "strength" : g.strength, # Definitely don't need this
            "output_type" : "tensor",
            "latents" : g.latents.to("cuda" if torch.cuda.is_available() else "cpu")
        }
        
        images = self.pipe(
            generator=generator,
            **settings
        ).images

        g.latents.to("cpu")

        # Convert to indices
        sample_indices = convert_to_level_format(images)
        
        # Add level data to the list
        scene = sample_indices[0].tolist() # Always just one scene: (1,16,16)
        #print(scene)
        g.scene = scene 
        if args.game == 'Mario':
            actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, self.args.describe_absence)
        elif args.game == 'MM2':
            actual_caption = self.mm_assign_caption(scene)
        elif args.game == 'LR':
            actual_caption = lr_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, self.args.describe_absence)
        # TODO: Generalize
        g.caption = actual_caption

        #print(f"Describe resulting image: {actual_caption}")
        #compare_score = compare_captions(self.prompt, actual_caption)
        #print(f"Comparison score: {compare_score}")

        if args.game == 'Mario':
            samples = visualize_samples(images)
        elif args.game == 'MM2':
            # game='MM2' renders the real MM2 sprites instead of flat Mario tiles
            samples = visualize_samples(images, game='MM2')
        elif args.game == 'LR':
            samples = visualize_samples(images, game='LR')
        # TODO: Generalize
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
        choices=["Mario", "MM2", "LR", "MM-Simple", "MM-Full", "MMLV"],
        help="Which game to create a model for (affects sample style and tile count)"
    )

    return parser.parse_args()

if __name__ == "__main__": 
    args = parse_args()

    config = common_settings.get_game_config(args.game)
    
    args.tileset_path = config["tileset"]
    args.width = config["width"]
        

    evolver = DiffusionEvolver(args.model_path, args.width, args.tileset_path, args=args)
    evolver.start_evolution()