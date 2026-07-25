from evolution.evolution import Evolver
from level_dataset import visualize_samples, convert_to_level_format
from create_ascii_captions import extract_tileset
import argparse
import torch
from evolution.genome import LatentGenome, disable_width_mutation
import util.common_settings as common_settings
from models.pipeline_loader import get_pipeline


class TextDiffusionEvolver(Evolver):
    def __init__(self, model_path, width, tileset_path=common_settings.MARIO_TILESET, args=None):
        Evolver.__init__(self, args)
        self.args = args
        self.width = width
        self.tileset_path = tileset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = get_pipeline(model_path).to(self.device)
        # Set negative prompt support in viewer if available
        self.negative_prompt_supported = getattr(self.pipe, "supports_negative_prompt", False)

        _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)

        self.config = common_settings.get_game_config(args.game) if args is not None else None
        if self.config["is_megaman"]:
            # Mega Man requires widths (and heights) that fit in a grid in a particular way.
            disable_width_mutation()
        self.caption_fn, self.compare_fn = self._build_caption_tools()
        # visualize_samples(images) defaults to Mario-style rendering, so only pass an
        # explicit game kwarg for everything else.
        self.render_kwargs = {} if self.config["cli_name"] == "Mario" else {"game": self.config["render_name"]}

    def _build_caption_tools(self):
        """Returns (caption_fn, compare_fn) for the current game, built centrally in
        common_settings.get_caption_tools (shared with ascii_data_browser.py's
        TileViewer and interactive_tile_level_generator.py's CaptionBuilder) so the
        per-game assign_caption/compare_captions wiring lives in one place instead
        of being re-listed in every script that needs it."""
        return common_settings.get_caption_tools(
            self.args.game,
            id_to_char=self.id_to_char,
            char_to_id=self.char_to_id,
            tile_descriptors=self.tile_descriptors,
            describe_absence=self.args.describe_absence,
            tileset_path=self.tileset_path,
        )

    def random_latent(self, seed=1):
        # Create the initial noise latents (this is what the pipeline does internally)
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
        self.genomes = [LatentGenome(self.width, seed, self.steps, self.guidance_scale, latents=self.random_latent(seed), prompt=self.prompt, negative_prompt=self.negative_prompt, num_segments=1) for seed in range(self.population_size)]
        self.viewer.id_to_char = self.id_to_char
        # char_to_id and tile_descriptors are needed by ImageGridViewer.edit_composed_scene
        # (passed straight through to LevelEditor) and by _astar_path_for_scene, both used
        # by the "Build Mega Man Level" button / MegaManLayoutEditor.
        self.viewer.char_to_id = self.char_to_id
        self.viewer.tile_descriptors = self.tile_descriptors

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

        # Mega Man scenes reserve void rows at the top that scale with the requested
        # height, chopped off before conversion to tile indices -- same chop_rows
        # formula as CaptionBuilder.generate_image (interactive_tile_level_generator.py).
        if self.config["is_megaman"]:
            chop_rows = (self.config["height"] // 16) * 2
            if chop_rows > 0:
                images = images[:, :, chop_rows:, :]

        # Convert to indices
        sample_indices = convert_to_level_format(images)

        # Add level data to the list
        scene = sample_indices[0].tolist()  # Always just one scene: (1,16,16)
        g.scene = scene
        g.caption = self.caption_fn(scene)

        # Score how well the generated scene matches the prompt it was conditioned
        # on: compare the prompt against the scene's own deterministic caption.
        if g.prompt and g.prompt.strip() and self.compare_fn is not None:
            g.score = self.compare_fn(g.prompt, g.caption)
        else:
            g.score = None
        print(f"Caption adherence score: {g.score}")

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
        choices=["Mario", "MM2", "LR", "MM-Simple", "MM-Full"],
        help="Which game to create a model for (affects sample style and tile count)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = common_settings.get_game_config(args.game)

    args.tileset_path = config["tileset"]
    args.width = config["width"]

    evolver = TextDiffusionEvolver(args.model_path, args.width, args.tileset_path, args=args)
    allow_negative_prompt = getattr(evolver.pipe, "supports_negative_prompt", False)
    evolver.start_evolution(allow_prompt=True, allow_negative_prompt=allow_negative_prompt)