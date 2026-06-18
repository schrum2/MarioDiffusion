from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from models.latent_diffusion_pipeline import UnconditionalDDPMPipeline
from models.fdm_pipeline import FDMPipeline
import os
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


def _latest_checkpoint_with_unet(model_path):
    """Return the highest-numbered ``checkpoint-N`` subdir that holds a saved unet, or None.

    Diffusion training writes weights into ``checkpoint-N`` subdirs and only optionally copies
    a final model to the top level. When that top-level copy is missing (e.g. a run that was
    stopped early, or a model shared without its final save) we can still load it from its most
    recent checkpoint.
    """
    if not os.path.isdir(model_path):
        return None
    best = None
    for name in os.listdir(model_path):
        full = os.path.join(model_path, name)
        if not (name.startswith("checkpoint-") and os.path.isdir(full)):
            continue
        if not os.path.exists(os.path.join(full, "unet")):
            continue
        try:
            num = int(name.split("-")[-1])
        except ValueError:
            continue
        if best is None or num > best[0]:
            best = (num, full)
    return best[1] if best else None


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
        elif os.path.exists(os.path.join(model_path, "config.json")):
            #New FDM saving
            pipe = FDMPipeline.from_pretrained(model_path)
        else:
            # No top-level diffusion or FDM save. If this is a diffusion model whose weights only
            # live in checkpoint subdirs, load the latest checkpoint so the original model dir
            # still works as --model_path. Otherwise the directory has no model at all (a common
            # cause: a typo'd --model_path, or an output dir auto-created before this load), so
            # fail with a clear message instead of a cryptic missing-config.json error.
            latest_checkpoint = _latest_checkpoint_with_unet(model_path)
            if latest_checkpoint is not None:
                pipe = get_pipeline(latest_checkpoint)
            else:
                raise FileNotFoundError(
                    f"'{model_path}' is not a loadable model directory. Expected a top-level "
                    f"'unet/' (diffusion), 'final-model/' or 'config.json' (FDM), or a "
                    f"'checkpoint-*/unet' subdirectory. Found: {sorted(os.listdir(model_path))}."
                )
    else:
        # Hugging Face Hub model: 
        # Need to generalize to support other pipelines
        pipe = TextConditionalDDPMPipeline.from_pretrained(model_path)

    return pipe