import argparse
import torch
from evolution.evolution import Evolver
from evolution.genome import LatentGenome, disable_width_mutation
from level_dataset import visualize_samples, convert_to_level_format
from create_ascii_captions import extract_tileset
from models.wgan_model import WGAN_Generator
from run_wgan import generate_level_scene_from_latent
import util.common_settings as common_settings


class WGANEvolver(Evolver):
    def __init__(self, args):
        Evolver.__init__(self, args)
        self.args = args
        self.width = args.width
        self.tileset_path = args.tileset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = common_settings.get_game_config(args.game) if args is not None else None
        if self.config and self.config.get("is_megaman"):
            disable_width_mutation()

        isize = self.config["height"] if self.config else args.width
        self.netG = WGAN_Generator(
            isize, args.nz, args.num_tiles, args.ngf, n_extra_layers=args.n_extra_layers
        )

        # Load trained model
        try:
            self.netG.load_state_dict(torch.load(args.model_path, map_location=self.device))
            print(f"Successfully loaded generator model from {args.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise ValueError(f"Failed to load model from {args.model_path}")

        # Move model to device and set to evaluation mode
        self.netG = self.netG.to(self.device)
        self.netG.eval()

        _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(self.tileset_path)

        self.caption_fn, self.compare_fn = self._build_caption_tools()

        # visualize_samples(images) defaults to Mario-style rendering, so only pass an
        # explicit game kwarg for everything else.
        self.render_kwargs = {} if self.config["cli_name"] == "Mario" else {"game": self.config["render_name"]}

    def _build_caption_tools(self):
        """Returns (caption_fn, compare_fn) built centrally in common_settings."""
        return common_settings.get_caption_tools(
            self.args.game,
            id_to_char=self.id_to_char,
            char_to_id=self.char_to_id,
            tile_descriptors=self.tile_descriptors,
            describe_absence=self.args.describe_absence,
            tileset_path=self.tileset_path,
        )

    def random_latent(self, seed=1, batch_size=1):
        # Set generator seed for reproducible latent vectors if specified
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        noise = torch.randn(batch_size, self.args.nz, 1, 1, generator=generator, device="cpu")
        return noise

    def initialize_population(self):
        self.genomes = [
            LatentGenome(
                self.width,
                seed,
                self.steps,
                self.guidance_scale,
                latents=self.random_latent(seed),
                num_segments=1,
            )
            for seed in range(self.population_size)
        ]

        self.viewer.id_to_char = self.id_to_char
        self.viewer.char_to_id = self.char_to_id
        self.viewer.tile_descriptors = self.tile_descriptors

    def generate_image(self, g):
        print(f"Generate new image for {g}")

        noise = g.latents.to(self.device)
        samples_cpu = generate_level_scene_from_latent(self.netG, noise)
        g.latents.to("cpu")

        # Mega Man scenes reserve void rows at the top that scale with height
        if self.config.get("is_megaman"):
            chop_rows = (self.config["height"] // 16) * 2
            if chop_rows > 0:
                samples_cpu = samples_cpu[:, :, chop_rows:, :]

        sample_indices = convert_to_level_format(samples_cpu)

        # Process generated level scene data
        scene = sample_indices[0].tolist()
        g.scene = scene

        if self.caption_fn:
            g.caption = self.caption_fn(scene)

        samples = visualize_samples(samples_cpu, **self.render_kwargs)
        return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Evolve levels with WGAN")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained WGAN model")
    parser.add_argument("--tileset_path", default=common_settings.MARIO_TILESET, help="Descriptions of individual tile types")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--width", type=int, default=common_settings.MARIO_WIDTH, help="Tile width of generated level")
    parser.add_argument("--num_tiles", type=int, default=common_settings.MARIO_TILE_COUNT, help="Number of tile types")

    parser.add_argument("--nz", type=int, default=32, help="Size of the latent z vector")
    parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--n_extra_layers", type=int, default=0, help="Number of extra layers in generator")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        choices=["Mario", "LR", "MM-Simple", "MM-Full", "MMLV", "MM2"],
        help="Which game to create a model for (affects sample style and tile count)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = common_settings.get_game_config(args.game)

    args.num_tiles = config["tile_count"]
    args.tileset_path = config["tileset"]
    args.width = config["width"]

    evolver = WGANEvolver(args)
    evolver.start_evolution()