import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import random
import numpy as np
from accelerate import Accelerator
from level_dataset import LevelDataset, visualize_samples
from tokenizer import Tokenizer 
import json
import threading
from datetime import datetime
from util.plotter import Plotter
from models.text_model import TransformerModel
from evaluate_caption_adherence import calculate_caption_score_and_samples
from create_ascii_captions import extract_tileset
from transformers import AutoTokenizer, AutoModel
from models.fdm import Gen
from models.fdm_pipeline import FDMPipeline
import util.common_settings as common_settings
import models.general_training_helper as gen_train_help
import models.sentence_transformers_helper as st_helper
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--pkl", type=str, default=None, help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="datasets\\SMB1_LevelsAndCaptions-regular-train.json", help="Path to dataset json file")
    parser.add_argument("--val_json", type=str, default=None, help="Optional path to validation dataset json file")
    parser.add_argument("--num_tiles", type=int, default=common_settings.MARIO_TILE_COUNT, help="Number of tile types")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument('--split', action='store_true', help='Enable train/val/test split') # TODO: Allow SMB1 data to be split into groups for training and testing
    
    # New text conditioning args
    parser.add_argument("--pretrained_language_model", type=str, default=None, help="Link to a pre-trained language model, everything after huggingface.co/.")
    
    # Model args
    parser.add_argument("--embedding_dim", type=int, default=384, help="Base size of the embedded tokens into the model")#TODO: not sure if this can be changed
    parser.add_argument("--z_dim", type=int, default=5, help="Size of the concatenated noise vector for varitey (Default 5)")
    parser.add_argument("--kern_size", type=int, default=7, help="Kernel size for convolutional layers (default 7)")
    parser.add_argument("--filter_count", type=int, default=128, help="Number of filters in the convolutional layers (default 128)")
    parser.add_argument("--num_res_blocks", type=int, default=3, help="Number of residual blocks in the generator (default 3)")


    # Training args
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size") # TODO: Consider reducing to 16 to help generalization
    parser.add_argument("--save_image_epochs", type=int, default=10, help="Save generated levels every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate_epochs", type=int, default=10, help="Calculate validation loss every N epochs")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="level-fdm-output", help="Output directory")

    # For caption score calculation
    parser.add_argument("--tileset", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--plot_validation_caption_score", action="store_true", default=False, help="Whether validation caption score should be plotted")

    # For block2vec embedding model
    parser.add_argument("--block_embedding_model_path", type=str, default=None, help="Path to trained block embedding model (.pt)")

    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file with training parameters.")


    return parser.parse_args()



class imageDataSet(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self,idx):
        return self.data[0][idx], self.data[1][idx], self.data[2][idx]




def main():
    args = parse_args()

    # Check if config file is provided before training loop begins
    if hasattr(args, 'config') and args.config:
        config = gen_train_help.load_config_from_json(args.config)
        args = gen_train_help.update_args_from_config(args, config)
        print("Training will use parameters from the config file.")


    # Check if output directory already exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print(f"Output directory '{args.output_dir}' already exists. Please remove it or choose a different name.")
        exit()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Setup accelerator
    accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
        )

    # Initialize tokenizer
    if args.pkl:
        tokenizer = Tokenizer()
        tokenizer.load(args.pkl)
    else:
        tokenizer = None

    # Load text embedding model if text conditioning is enabled
    if args.pretrained_language_model: #Default to huggingface model, if it exists
        text_encoder = AutoModel.from_pretrained(args.pretrained_language_model, trust_remote_code=True).to(accelerator.device)
        text_encoder.eval() # Set to evaluation mode
        model_embedding_dim = text_encoder.config.hidden_size#TODO: not sure if this can be changed
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_language_model)
        print(f"Loaded text encoder from {args.pretrained_language_model}")
    else:
        raise ValueError("You must provice a pretrained text embedding model!")
    

    # Load block embedding model if specified
    block_embeddings = None
    embedding_dim = None
    if args.block_embedding_model_path:
        try:
            # Load embeddings from the embeddings.pt file in the model directory
            block_embeddings = torch.load(
                os.path.join(args.block_embedding_model_path, "embeddings.pt"),
                map_location=accelerator.device
            )
            
            embedding_dim = block_embeddings.shape[1]
            print(f"Loaded block embeddings from {args.block_embedding_model_path} with dimension {embedding_dim}")
        except Exception as e:
            print(f"Error loading block embedding model: {e}")
            raise
    

    # Initialize dataset
    data_mode = "diff_text"
    train_dataloader, val_dataloader = gen_train_help.create_dataloaders(json_path=args.json,
                                        val_json=args.val_json, tokenizer=tokenizer, data_mode=data_mode,
                                        augment=args.augment, num_tiles=args.num_tiles,
                                        negative_prompt_training=False,
                                        block_embeddings=block_embeddings, batch_size=args.batch_size)


    sample_captions, _ = gen_train_help.get_random_training_samples(train_dataloader, False, args.output_dir)
    

    #Create an instance of the model
    model = Gen(
        model_name="Five-Dollar-Model",
        embedding_dim=args.embedding_dim,
        z_dim=args.z_dim,
        kern_size=args.kern_size,
        filter_count=args.filter_count,
        num_res_blocks=args.num_res_blocks,
        out_channels=embedding_dim if args.block_embedding_model_path else args.num_tiles
    )
    model.to(accelerator.device)
    
    

    # Training loop
    global_step = 0
    progress_bar = tqdm(total=args.num_epochs * len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    # Get formatted timestamp for filenames
    formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')

    # Create log files
    log_file = os.path.join(args.output_dir, f"training_log_{formatted_date}.jsonl")
    config_file = os.path.join(args.output_dir, f"hyperparams_{formatted_date}.json")

    # Save hyperparameters to JSON file
    if accelerator.is_local_main_process:
        hyperparams = vars(args)
        with open(config_file, "w") as f:
            json.dump(hyperparams, f, indent=4)
        print(f"Saved configuration to: {config_file}")


    # Initialize plotter if we're on the main process
    plotter, plot_thread = None, None

    caption_score_plotter, caption_score_plot_thread = None, None
    
    caption_score_log_file = os.path.join(args.output_dir, f"caption_score_log_{formatted_date}.jsonl")

    if accelerator.is_local_main_process:
        plotter, plot_thread = gen_train_help.start_plotter(log_file=log_file, output_dir=args.output_dir,
                                            left_key='loss', right_key='val_loss', left_label='Training Loss', 
                                            right_label='Validation Loss', png_name='training_loss')
        
        caption_score_plotter = None
        if args.plot_validation_caption_score:
            # Caption score plotter
            caption_score_plotter, caption_score_plot_thread = gen_train_help.start_plotter(
                                            log_file=caption_score_log_file, output_dir=args.output_dir,
                                            left_key='caption_score', right_key=None, left_label='Caption Match Score', 
                                            right_label=None, png_name='caption_score')
            
            _, id_to_char, char_to_id, tile_descriptors = extract_tileset(args.tileset)
    


    optimizer = torch.optim.Adam(model.parameters())
    





    formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')
    log_file = os.path.join(args.output_dir, f"training_log_{formatted_date}.jsonl")
    config_file = os.path.join(args.output_dir, f"hyperparams_{formatted_date}.json")

    def log_metrics(epoch, loss, step=None, val_loss=None):
        if accelerator.is_local_main_process:
            log_entry = {
                "epoch": epoch,
                "loss": loss,
                "step": step if step is not None else epoch * len(train_dataloader),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if val_loss is not None:
                log_entry["val_loss"] = val_loss
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    global_step = 0

    #Used for finding the best model to save
    best_model_epoch=0
    best_caption_score=-1


    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                
                optimizer.zero_grad()

                loss = process_fdm_batch(
                    args, model, batch, tokenizer, text_encoder, accelerator.device)
                
                accelerator.backward(loss)                
                optimizer.step()
                
            
            train_loss += loss.detach().item()
            # Update progress bar
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
                        
            global_step += 1

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_dataloader)


        # Calculate validation loss if validation dataset exists and it's time to validate
        val_loss = None
        avg_caption_score = None
        if val_dataloader is not None and (epoch % args.validate_epochs == 0 or epoch == args.num_epochs - 1):
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_batch_loss = process_fdm_batch(
                        args, model, val_batch, tokenizer, text_encoder, accelerator.device
                    )
                    val_loss += val_batch_loss.item()
            val_loss /= len(val_dataloader)

            if args.plot_validation_caption_score:
                # Compute caption match score for this data
                pipeline = FDMPipeline(
                    tokenizer, text_encoder, model, accelerator.device
                ).to(accelerator.device)
                # Only use the positive captions for scoring

                # TODO: These should be argparse parameters
                avg_caption_score, _, _, _ = calculate_caption_score_and_samples(
                    accelerator.device, pipeline, val_dataloader, None, None, args.seed,
                    id_to_char=id_to_char, char_to_id=char_to_id, tile_descriptors=tile_descriptors, 
                    describe_absence=args.describe_absence, output=False, height=16, width=16
                )

                if avg_caption_score>best_caption_score:
                    best_model_epoch=epoch
                    best_caption_score=avg_caption_score
            else:
                # Is this how this should behave in the unconditional case?
                # Or should I justs use 0 or -1?
                avg_caption_score = None

            model.train()
            # Log caption match score
            if accelerator.is_local_main_process:
                with open(caption_score_log_file, 'a') as f:
                    log_entry = {
                        "epoch": epoch,
                        "caption_score": avg_caption_score,                
                        "step": global_step,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    f.write(json.dumps(log_entry) + '\n')
        
        # Log metrics including validation loss
        log_metrics(epoch, avg_train_loss, val_loss=val_loss, step=global_step)
        
        # Generate and save sample levels every N epochs
        if epoch % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            # Switch to eval mode
            model.eval()
            

            pipeline = FDMPipeline(
                tokenizer, text_encoder, model, accelerator.device
            ).to(accelerator.device)
                            
            # Use the raw negative captions instead of tokens
            with torch.no_grad():
                samples = pipeline(
                    batch_size=4,
                    generator=torch.Generator(device=accelerator.device).manual_seed(args.seed),
                    caption=sample_captions,
                    show_progress_bar=False,
                ).images

            # Convert one-hot samples to tile indices and visualize
            prompts = sample_captions
            visualize_samples(samples, os.path.join(args.output_dir, f"samples_epoch_{epoch}"), prompts=prompts)
            
        # Save model every N epochs
        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            # Save the model
            pipeline = FDMPipeline(
            tokenizer, text_encoder, model, accelerator.device
            ).to(accelerator.device)
                
            pipeline.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch}"))
        
    try:
        # Clean up plotting resources
        if accelerator.is_local_main_process and plotter:
            # Better thread cleanup
            gen_train_help.kill_plotter(plotter=plotter, plot_thread=plot_thread)

            gen_train_help.kill_plotter(plotter=caption_score_plotter, plot_thread=caption_score_plot_thread)

        # Force CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Ensure all processes are synchronized
        accelerator.wait_for_everyone()

    finally:
        # Close progress bar and TensorBoard writer
        progress_bar.close()

        print(f"Saving the best model at epoch {best_model_epoch}")
        #Copy the best model into the main directory
        shutil.copytree(os.path.join(args.output_dir, f"checkpoint-{best_model_epoch}"), args.output_dir, dirs_exist_ok=True)

        



def process_fdm_batch(args, model, batch, tokenizer, text_encoder, device):
    #Gets and returns the loss of the model for a given batch
    scenes, captions = batch

    encoded_captions = prepare_conditioned_batch(tokenizer, text_encoder, captions, device) #get text embeds

    #encoded_captions.shape[0] to get the same number of noise vectors as encoded captions to match the batches
    noise = torch.randn(encoded_captions.shape[0], args.z_dim, device=device)

    predicted = model(encoded_captions, noise) #Model output

    loss = torch.nn.NLLLoss()(torch.log(predicted), scenes.argmax(dim=1).to(device)) #Get the loss here

    return loss


def prepare_conditioned_batch(tokenizer_hf, text_encoder, captions, device):
    #Returns the text embeddings for a given caption
    with torch.no_grad():         
        combined_embeddings = st_helper.encode(texts=captions, tokenizer=tokenizer_hf,
                                                model=text_encoder, device=device)
        combined_embeddings = combined_embeddings*6 #Multiply by a scaling factor, this helps prevent errors later
    return combined_embeddings




if __name__ == "__main__":
    main()
