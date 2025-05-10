from evolution.evolution import Evolver
from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples, convert_to_level_format
from create_ascii_captions import extract_tileset
import argparse
import torch
from evolution.genome import LatentGenome
from create_ascii_captions import assign_caption

class TextDiffusionEvolver(Evolver):
    def __init__(self, model_path, width, tileset_path='..\TheVGLC\Super Mario Bros\smb.json', args = None):
        Evolver.__init__(self)

        self.args = args
        self.width = width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = TextConditionalDDPMPipeline.from_pretrained(model_path).to(self.device)

        _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)

    def random_latent(self, seed=1):
        # Create the initial noise latents (this is what the pipeline does internally)
        height = 16
        width = self.width
        num_channels_latents = len(self.id_to_char)
        latents_shape = (1, num_channels_latents, height, width)
        latents = torch.randn(
            latents_shape, 
            generator=torch.manual_seed(seed)        
        ).to("cpu")
        return latents

    def initialize_population(self):
        self.genomes = [LatentGenome(self.width, seed, self.steps, self.guidance_scale, latents=self.random_latent(seed), prompt=self.prompt, num_segments=1, generation_width=16) for seed in range(self.population_size)]
        self.viewer.id_to_char = self.id_to_char

    def generate_image(self, g):
        # generate fresh new image
        print(f"Generate new image for {g}")
        generator = torch.Generator("cuda").manual_seed(g.seed)

        settings = {
            "batch_size" : 1,
            "guidance_scale" : g.guidance_scale, 
            "num_inference_steps" : g.num_inference_steps,
            # "strength" : g.strength, # Definitely don't need this
            "output_type" : "tensor",
            "raw_latent_sample" : g.latents.to("cuda")
        }

        # Include caption if desired
        prompt = g.prompt
        if prompt.strip() != "":
            sample_captions = [prompt] # batch of size 1
            sample_caption_tokens = self.pipe.text_encoder.tokenizer.encode_batch(sample_captions)
            sample_caption_tokens = torch.tensor(sample_caption_tokens).to(self.device)

            settings["captions"] = sample_caption_tokens
        
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

        actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, self.args.describe_locations, self.args.describe_absence)
        g.caption = actual_caption

        #print(f"Describe resulting image: {actual_caption}")
        #compare_score = compare_captions(self.prompt, actual_caption)
        #print(f"Comparison score: {compare_score}")

        return visualize_samples(images)

def parse_args():
    parser = argparse.ArgumentParser(description="Evolve levels with unconditional diffusion model")    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--tileset_path", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--width", type=int, default=16, help="Tile width of generated level")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evolver = TextDiffusionEvolver(args.model_path, args.width, args.tileset_path, args=args)
    evolver.start_evolution(allow_prompt = True)