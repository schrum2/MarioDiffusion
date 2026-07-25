import argparse
import torch
from evolution.evolution import Evolver
from evolution.genome import LatentGenome, disable_width_mutation
from level_dataset import visualize_samples, convert_to_level_format
from create_ascii_captions import extract_tileset
from models.pipeline_loader import get_pipeline
import util.common_settings as common_settings


class DiffusionEvolver(Evolver):
    def __init__(self, model_path, width, tileset_path=common_settings.MARIO_TILESET, args=None):
        Evolver.__init__(self, args)
        self.args = args
        self.width = width
        self.tileset_path = tileset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = get_pipeline(model_path).to(self.device)

        _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)

        self.config = common_settings.get_game_config(args.game) if args is not None else None
        if self.config and self.config.get("is_megaman"):
            # Mega Man requires widths and heights that fit in a grid in a particular way
            disable_width_mutation()

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

    def random_latent(self, seed=1):
        # Create initial noise latents matching UNet input dimensions
        height = self.config["height"]
        width = self.width
        num_channels_latents = self.pipe.unet.config.in_channels

        latents_shape = (1, num_channels_latents, height, width)
        latents = torch.randn(
            latents_shape, 
            generator=torch.manual_seed(seed)
        ).to("cpu")
        return latents

    def initialize_population(self):
        self.genomes = [
            LatentGenome(
                self.width, 
                seed, 
                self.steps, 
                self.guidance_scale, 
                latents=self.random_latent(seed), 
                num_segments=1
            ) 
            for seed in range(self.population_size)
        ]
        
        self.viewer.id_to_char = self.id_to_char
        self.viewer.char_to_id = self.char_to_id
        self.viewer.tile_descriptors = self.tile_descriptors

    def generate_image(self, g):
        print(f"Generate new image for {g}")
        generator = torch.Generator(self.device.type).manual_seed(g.seed)

        settings = {
            "batch_size": 1,
            "num_inference_steps": g.num_inference_steps,
            "output_type": "tensor",
            "latents": g.latents.to(self.device)
        }

        images = self.pipe(
            generator=generator,
            **settings
        ).images

        g.latents.to("cpu")

        # Mega Man scenes reserve void rows at the top that scale with the requested
        # height, chopped off before conversion to tile indices.
        if self.config.get("is_megaman"):
            chop_rows = (self.config["height"] // 16) * 2
            if chop_rows > 0:
                images = images[:, :, chop_rows:, :]

        # Convert to indices
        sample_indices = convert_to_level_format(images)

        # Process generated level data
        scene = sample_indices[0].tolist()
        g.scene = scene 
        
        if self.caption_fn:
            g.caption = self.caption_fn(scene)

        samples = visualize_samples(images, **self.render_kwargs)
        return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Evolve levels with unconditional diffusion model")
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--tileset_path", default=common_settings.MARIO_TILESET, help="Descriptions of individual tile types")
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