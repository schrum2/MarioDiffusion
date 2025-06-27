from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from models.latent_diffusion_pipeline import UnconditionalDDPMPipeline
from models.fdm_pipeline import FDMPipeline
import os
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


def get_pipeline(model_path):
    # If model_path is a local directory, use the original logic
    if os.path.isdir(model_path):
        #Diffusion models
        if os.path.exists(os.path.join(model_path, "unet")):
            if os.path.exists(os.path.join(model_path, "text_encoder")):
                #If it has a text encoder and a unet, it's text conditional diffusion
                pipe = TextConditionalDDPMPipeline.from_pretrained(model_path)
            else:
                #If it has no text encoder, use the unconditional diffusion model
                pipe = UnconditionalDDPMPipeline.from_pretrained(model_path)
        #Get the FDM pipeline if "unet" doesn't exist
        elif os.path.exists(os.path.join(model_path, "final-model")):
            #Legacy FDM saving
            pipe = FDMPipeline.from_pretrained(os.path.join(model_path, "final-model"))
        else:
            #New FDM saving
            pipe = FDMPipeline.from_pretrained(model_path)
    else:
        # Assume it's a Hugging Face Hub model ID
        # Try to load config to determine if it's text-conditional
        try:
            config, _ = DiffusionPipeline.load_config(model_path)
            components = config.get("components", {})
        except Exception:
            components = {}
        if "text_encoder" in components or "text_encoder" in str(components):
            # Use the local pipeline file for custom_pipeline
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                custom_pipeline="models.text_diffusion_pipeline.TextConditionalDDPMPipeline",
                trust_remote_code=True,
            )
        else:
            # Fallback: try unconditional 
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                custom_pipeline="models.latent_diffusion_pipeline.UnconditionalDDPMPipeline",
                trust_remote_code=True,
            )

    return pipe