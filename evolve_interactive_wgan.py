from evolution.evolution import Evolver
from level_dataset import visualize_samples, convert_to_level_format
from create_ascii_captions import extract_tileset
import argparse
import torch
from models.wgan_model import WGAN_Generator
from run_wgan import generate_level_scene_from_latent
from evolution.genome import LatentGenome, disable_width_mutation
from create_ascii_captions import assign_caption
from LR_create_ascii_captions import assign_caption as lr_assign_caption
import util.common_settings as common_settings

class WGANEvolver(Evolver):
    def __init__(self, args):
        Evolver.__init__(self, args)

        self.args = args

        self.width = args.width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG = WGAN_Generator(isize, args.nz, args.num_tiles, args.ngf, n_extra_layers=args.n_extra_layers)

        # Load trained model
        try:
            self.netG.load_state_dict(torch.load(args.model_path, map_location=self.device))
            print(f"Successfully loaded generator model from {args.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise ValueError(f"failed {args.model_path}")

        # Move model to device and set to evaluation mode
        self.netG = self.netG.to(self.device)
        self.netG.eval()

        _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(args.tileset_path)

    def random_latent(self, seed=1, batch_size=1):
        # Generate random noise
        noise = torch.randn(batch_size, self.args.nz, 1, 1, device="cpu") # On CPU since population evolves in memory
        return noise

    def initialize_population(self):
        self.genomes = [LatentGenome(self.width, seed, self.steps, self.guidance_scale, latents=self.random_latent(seed), num_segments=1) for seed in range(self.population_size)]
        self.viewer.id_to_char = self.id_to_char

    def generate_image(self, g):
        # generate fresh new image
        print(f"Generate new image for {g}")

        noise = g.latents.to(self.device)
        samples_cpu = generate_level_scene_from_latent(self.netG, noise)
        g.latents.to("cpu")

        #print(samples_cpu)
        #print(samples_cpu.shape)
        sample_indices = convert_to_level_format(samples_cpu)
        #print(sample_indices)
        
        # Add level data to the list
        scene = sample_indices[0].tolist() # Always just one scene: (1,16,16)
        #print(scene)
        g.scene = scene 

        if args.game == 'Mario':
            actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, self.args.describe_absence)
        elif args.game == 'LR':
            actual_caption = lr_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, self.args.describe_absence)
        g.caption = actual_caption

        #print(f"Describe resulting image: {actual_caption}")
        #compare_score = compare_captions(self.prompt, actual_caption)
        #print(f"Comparison score: {compare_score}")

        if args.game == 'Mario':
            samples = visualize_samples(samples_cpu)
        elif args.game == 'LR':
            samples = visualize_samples(samples_cpu, game='LR')
        return samples

def parse_args():
    parser = argparse.ArgumentParser(description="Evolve levels with WGAN")    

    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--tileset_path", default=common_settings.MARIO_TILESET, help="Descriptions of individual tile types")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--width", type=int, default=common_settings.MARIO_WIDTH, help="Tile width of generated level")
    parser.add_argument("--num_tiles", type=int, default=common_settings.MARIO_TILE_COUNT, help="Number of tile types")
    

    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    ### parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--nz", type=int, default=32, help="Size of the latent z vector")
    parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--n_extra_layers", type=int, default=0, help="Number of extra layers in generator")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        choices=["Mario", "LR", "MM-Simple", "MM-Full"],
        help="Which game to create a model for (affects sample style and tile count)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    disable_width_mutation()

    if args.game == "Mario":
        args.num_tiles = common_settings.MARIO_TILE_COUNT
        isize = common_settings.MARIO_HEIGHT
        args.width = common_settings.MARIO_WIDTH
        args.tileset_path = common_settings.MARIO_TILESET
    elif args.game == "LR":
        args.num_tiles = common_settings.LR_TILE_COUNT
        isize = common_settings.LR_HEIGHT
        args.width = common_settings.LR_WIDTH
        args.tileset_path = common_settings.LR_TILESET
    elif args.game == "MM-Simple":
        args.num_tiles = common_settings.MM_SIMPLE_TILE_COUNT
        isize = common_settings.MEGAMAN_HEIGHT      # Assuming square samples
        args.width = common_settings.MEGAMAN_WIDTH  # Assuming square samples
    elif args.game == "MM-Full":
        args.num_tiles = common_settings.MM_FULL_TILE_COUNT
        isize = common_settings.MEGAMAN_HEIGHT      # Assuming square samples
        args.width = common_settings.MEGAMAN_WIDTH  # Assuming square samples
    else:
        raise ValueError(f"Unknown game: {args.game}")
    
    evolver = WGANEvolver(args)
    evolver.start_evolution()
