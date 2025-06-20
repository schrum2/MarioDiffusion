from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from models.latent_diffusion_pipeline import UnconditionalDDPMPipeline
from models.fdm_pipeline import FDMPipeline
import os

def get_pipeline(model_path):

    #Diffusion models
    if(os.path.exists(os.path.join(model_path, "unet"))):

        if(os.path.exists(os.path.join(model_path, "text_encoder"))):
            #If it has a text encoder and a unet, it's text conditional diffusion
            pipe = TextConditionalDDPMPipeline.from_pretrained(model_path)
        else:
            #If it has no text encoder, use the unconditional diffusion model
            pipe = UnconditionalDDPMPipeline.from_pretrained(model_path)
    
    #Get the FDM pipeline if "unet" doesn't exist
    elif(os.path.exists(os.path.join(model_path, "final-model"))):
        #Legacy FDM saving
        pipe = FDMPipeline.from_pretrained(os.path.join(model_path, "final-model"))
    else:
        #New FDM saving
        pipe = FDMPipeline.from_pretrained(model_path)
    
    return pipe