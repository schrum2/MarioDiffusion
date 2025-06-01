import argparse
import os
import torch
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup 
from tqdm.auto import tqdm
import random
import numpy as np
from accelerate import Accelerator
from level_dataset import visualize_samples
from tokenizer import Tokenizer 
import json
from datetime import datetime
from models.text_model import TransformerModel
from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from models.latent_diffusion_pipeline import UnconditionalDDPMPipeline
from evaluate_caption_adherence import calculate_caption_score_and_samples
from captions.util import extract_tileset # TODO: Move this to a caption_util.py file
from transformers import AutoTokenizer, AutoModel
import util.common_settings as common_settings
from torch.distributions import Categorical
from models.block2vec_model import Block2Vec
import models.sentence_transformers_helper as st_helper
import models.text_model as text_model
import glob
import models.general_training_helper as gen_train_help
import re

def mse_loss(pred, target, scene_oh=None, noisy_scenes=None, **kwargs):
    """Standard MSE loss between prediction and target."""
    return torch.nn.functional.mse_loss(pred, target)


def reconstruction_loss(pred, target, scene_oh, noisy_scenes, timesteps=None, scheduler=None, **kwargs):
    """
    Reconstruction loss using negative log-likelihood (cross-entropy) as in DDPM for categorical data.
    Args:
        pred: predicted noise, shape [batch, classes, H, W]
        scene_oh: original scene, one-hot, shape [batch, classes, H, W]
        noisy_scenes: x_t, shape [batch, classes, H, W]
        timesteps: [batch] (long tensor of timesteps for each sample)
        scheduler: DDPMScheduler instance (needed for alphas_cumprod)
    """
    if timesteps is None or scheduler is None:
        raise ValueError("timesteps and scheduler must be provided for reconstruction_loss")
    # Get alpha_hat for each sample in the batch
    alpha_hat = scheduler.alphas_cumprod[timesteps].to(pred.device)  # [batch]
    sqrt_alpha_hat = torch.sqrt(alpha_hat)[:, None, None, None]      # [batch, 1, 1, 1]
    sqrt_one_minus_alpha_hat = torch.sqrt(1. - alpha_hat)[:, None, None, None]  # [batch, 1, 1, 1]
    # Reconstruct logits for x_0 (original image)
    logits = (1.0 / sqrt_alpha_hat) * (noisy_scenes - sqrt_one_minus_alpha_hat * pred)  # [batch, classes, H, W]
    # Prepare targets as class indices
    target_indices = scene_oh.argmax(dim=1)  # [batch, H, W]
    # Categorical expects [batch, H, W, classes]
    logits = logits.permute(0, 2, 3, 1)  # [batch, H, W, classes]
    dist = Categorical(logits=logits)
    rec_loss = -dist.log_prob(target_indices).sum(dim=(1,2)).mean()
    return rec_loss


def combined_loss(pred, target, scene_oh=None, noisy_scenes=None, timesteps=None, scheduler=None, **kwargs):
    """Combined MSE and reconstruction loss."""
    mse = mse_loss(pred, target)
    rec = reconstruction_loss(pred, target, scene_oh, noisy_scenes, timesteps=timesteps, scheduler=scheduler)
    return mse + 0.001 * rec  # 0.001 can be made a parameter


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--pkl", type=str, default=None, help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="datasets\\SMB1_LevelsAndCaptions-regular-train.json", help="Path to dataset json file")
    parser.add_argument("--val_json", type=str, default=None, help="Optional path to validation dataset json file")
    parser.add_argument("--num_tiles", type=int, default=13, help="Number of tile types")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size") # TODO: Consider reducing to 16 to help generalization
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    
    # New text conditioning args
    parser.add_argument("--mlm_model_dir", type=str, default="mlm", help="Path to pre-trained text embedding model")
    parser.add_argument("--pretrained_language_model", type=str, default=None, help="Link to a pre-trained language model, everything after huggingface.co/. This will override the mlm_model_dir argument.")
    parser.add_argument("--text_conditional", action="store_true", help="Enable text conditioning")
    parser.add_argument("--negative_prompt_training", action="store_true", help="Enable training with negative prompts")
    parser.add_argument("--split_pretrained_sentences", action="store_true", default=False, help="Instead of encoding the whole prompt at once using the pretrained model, enable splitting the prompt into compoent sentences.")
    
    # Model args
    parser.add_argument("--model_dim", type=int, default=128, help="Base dimension of UNet model")
    parser.add_argument("--dim_mults", nargs="+", type=int, default=[1, 2, 4], help="Dimension multipliers for UNet")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per downsampling")
    parser.add_argument("--down_block_types", nargs="+", type=str, 
                       default=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"], 
                       help="Down block types for UNet")
    parser.add_argument("--up_block_types", nargs="+", type=str, 
                       default=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"], 
                       help="Up block types for UNet")
    parser.add_argument("--attention_head_dim", type=int, default=8, help="Number of attention heads")
    
    # Training args
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_warmup_percentage", type=float, default=0.05, help="Learning rate warmup portion") 
    parser.add_argument("--lr_scheduler_cycles", type=float, default=0.5, help="Number of cycles for the cosine learning rate scheduler")
    parser.add_argument("--save_image_epochs", type=int, default=20, help="Save generated levels every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=20, help="Save model every N epochs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate_epochs", type=int, default=5, help="Calculate validation loss every N epochs")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="level-diffusion-output", help="Output directory")
    parser.add_argument("--best_model_criterion",type=str,default="val_loss",choices=["val_loss", "caption_score"],help="Criterion to determine the best model: 'val_loss' for lowest validation loss, 'caption_score' for highest caption score")
    
    # Diffusion scheduler args
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--num_inference_timesteps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Number of diffusion timesteps during inference (samples, caption adherence)")
    parser.add_argument("--beta_schedule", type=str, default="linear", help="Beta schedule type")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Beta schedule start value")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta schedule end value")
    
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file with training parameters.")

    # For caption score calculation
    parser.add_argument("--tileset", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--plot_validation_caption_score", action="store_true", default=False, help="Whether validation caption score should be plotted")

    # For block2vec embedding model
    parser.add_argument("--block_embedding_model_path", type=str, default=None, help="Path to trained block embedding model (.pt)")

    # Allows for optional loss function: default is MSE and cross-entropy is the alternative
    parser.add_argument(
        "--loss_type",
        type=str,
        default="COMBO",
        choices=["MSE", "REC", "COMBO"],
        help="Loss function to use: 'MSE' for mean squared error (default), 'REC' for reconstuction loss, 'COMBO' for both (TODO: add weight parameter)",
    )

    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        choices=["Mario", "LR"],
        help="Which game to create a model for (affects sample style and tile count)"
    )

    parser.add_argument(
        "--sprite_temperature_n",
        type=int,
        default=None,
        help="If set, enables per-sprite temperature scaling with the specified n (e.g., 2, 4, 8) during inference."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Number of epochs to wait for improvement before early stopping."
    )

    return parser.parse_args()

# TODO: We'll probably want to move this somewhere else eventually
def compute_sprite_scaling_factors(json_path, num_tiles, n):
    """
    Computes per-sprite scaling factors for temperature scaling.
    Args:
        json_path (str): Path to your level JSON file.
        num_tiles (int): Number of tile types.
        n (int): The temperature scaling root (e.g., 2, 4, 8).
    Returns:
        torch.Tensor: Scaling factors of shape [num_tiles].
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    counts = [0] * num_tiles
    for entry in data:
        # Assumes entry['level'] is a 2D array of tile indices
        level = entry.get('level')
        if level is not None:
            for row in level:
                for tile in row:
                    counts[tile] += 1
    # Avoid division by zero for unused tiles
    counts = [c if c > 0 else 1 for c in counts]
    scalings = [c ** (1 / n) for c in counts]
    min_scaling = min(scalings)
    scalings = [s / min_scaling for s in scalings]
    return torch.tensor(scalings, dtype=torch.float32)

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory and extract its epoch number."""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None, None
    # Extract epoch numbers and find the max
    pattern = re.compile(r"checkpoint-(\d+)")
    epochs = [(int(pattern.search(os.path.basename(c)).group(1)), c) for c in checkpoints if pattern.search(os.path.basename(c))]
    if not epochs:
        return None, None
    latest_epoch, latest_ckpt = max(epochs, key=lambda x: x[0])
    return latest_ckpt, latest_epoch

def copy_log_up_to_epoch(output_dir, log_file, resume_epoch):
    """
    Find the most recent previous training log in output_dir (excluding log_file itself),
    and copy entries up to resume_epoch into log_file.
    """
    # Find all previous log files except the new one
    log_files = [
        f for f in glob.glob(os.path.join(output_dir, "training_log_*.jsonl"))
        if os.path.abspath(f) != os.path.abspath(log_file)
    ]
    if not log_files:
        print("No previous log file found to copy from.")
        exit()
    # Pick the most recent one by modification time
    prev_log_file = max(log_files, key=os.path.getmtime)
    print(f"Copying log entries from {prev_log_file} up to epoch {resume_epoch} into {log_file}")

    with open(prev_log_file, 'r') as fin, open(log_file, 'w') as fout:
        for line in fin:
            try:
                entry = json.loads(line)
                if entry.get("epoch", -1) <= resume_epoch:
                    fout.write(line)
            except Exception:
                print(f"Warning: Skipping a malformed log line in {prev_log_file}.")
                exit()
    print(f"Truncated log file {log_file} to only include entries up to epoch {resume_epoch}")

def infer_global_step_from_log(log_file):
    """
    Reads the last valid 'step' value from the log file.
    Returns 0 if the log is empty or no step is found.
    """
    global_step = 0
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "step" in entry:
                        global_step = entry["step"]
                except Exception:
                    continue
    except Exception:
        pass
    return global_step

def main():
    args = parse_args()

    """
        The following logic defines the loss function variable based on user input.
        Note: The model expects one-hot encoded targets for both loss types..
    """
    if args.loss_type == "MSE":
        loss_fn = mse_loss
    elif args.loss_type == "REC":
        loss_fn = reconstruction_loss
    elif args.loss_type == "COMBO":
        loss_fn = combined_loss
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    # Print the selected loss function to console
    print(f"Using loss function: {args.loss_type}")

    if args.game == "Mario":
        args.num_tiles = common_settings.MARIO_TILE_COUNT
        args.tileset = '..\TheVGLC\Super Mario Bros\smb.json'
    elif args.game == "LR":
        args.num_tiles = common_settings.LR_TILE_COUNT # TODO
        args.tileset = '..\TheVGLC\Lode Runner\Loderunner.json' # TODO
    else:
        raise ValueError(f"Unknown game: {args.game}")

    # Check if config file is provided before training loop begins
    if hasattr(args, 'config') and args.config:
        config = gen_train_help.load_config_from_json(args.config)
        args = gen_train_help.update_args_from_config(args, config)
        print("Training will use parameters from the config file.")

    # Check if output directory already exists
    if os.path.exists(args.output_dir):
        checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
        if checkpoints:
            user_input = input(f"Output directory '{args.output_dir}' already exists and contains checkpoints. Resume training from last checkpoint? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Exiting. Please remove the directory or choose a different output directory.")
                exit()
            resume_training = True
        else:
            print(f"Output directory '{args.output_dir}' already exists but contains no checkpoints. Please remove it or choose a different name.")
            exit()
    else:
        os.makedirs(args.output_dir)
        resume_training = False
    
    if args.negative_prompt_training and not args.text_conditional:
        raise ValueError("Negative prompt training requires text conditioning to be enabled")
    
    if args.split_pretrained_sentences and not args.pretrained_language_model:
        raise ValueError("Sentence splitting requires the use of a pretrained language model")
    
    """
    If sprite temperature scaling is enabled and the model is unconditional, 
    then compute the scaling factors.
    Note: Applying per-sprite temperature scaling could conflict with the intent of the prompt
    on conditional models. Thus, this argument is only for unconditional models.
    """
    sprite_scaling_factors = None
    if (not args.text_conditional) and (args.sprite_temperature_n is not None):
        raise ValueError("temperature scaling not currently implemented")
        sprite_scaling_factors = compute_sprite_scaling_factors(
            args.json, args.num_tiles, args.sprite_temperature_n
        )
        print(f"Sprite scaling factors: {sprite_scaling_factors}")


    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Initialize tokenizer
    if args.pkl:
        tokenizer = Tokenizer()
        tokenizer.load(args.pkl)
    else:
        tokenizer = None

    # Load text embedding model if text conditioning is enabled
    text_encoder = None
    tokenizer_hf = None #We don't need the huggingface tokenizer if we're using our own, varible initialization done to avoid future errors
    if args.text_conditional and args.pretrained_language_model: #Default to huggingface model, if it exists
        text_encoder = AutoModel.from_pretrained(args.pretrained_language_model, trust_remote_code=True).to(accelerator.device)
        text_encoder.eval() # Set to evaluation mode
        model_embedding_dim = text_encoder.config.hidden_size# Done here to allow for cross-functionality with the mlm model
        tokenizer_hf = AutoTokenizer.from_pretrained(args.pretrained_language_model)
        print(f"Loaded text encoder from {args.pretrained_language_model}")
    elif args.text_conditional and args.mlm_model_dir:
        text_encoder = TransformerModel.from_pretrained(args.mlm_model_dir).to(accelerator.device)
        text_encoder.eval()  # Set to evaluation mode
        model_embedding_dim = text_encoder.embedding_dim #Done to allow for cross-functionality with the huggingface model
        print(f"Loaded text encoder from {args.mlm_model_dir}")
    
    data_mode = "diff_text"

    # Load block embedding model if specified
    block_embeddings = None
    embedding_dim = None
    if args.block_embedding_model_path:
        try:
            block2vec = Block2Vec.from_pretrained(args.block_embedding_model_path)
            block_embeddings = block2vec.get_embeddings()
            embedding_dim = block_embeddings.shape[1]
            print(f"Loaded block embeddings from {args.block_embedding_model_path} with dimension {embedding_dim}")
            print("Block embedding model loaded successfully.")
        except Exception as e:
            print(f"Error loading block embedding model: {e}")
            raise
    else:
        print("No block embedding model specified. One-hot encoding enabled.")

    train_dataloader, val_dataloader = gen_train_help.create_dataloaders(json_path=args.json,
                                        val_json=args.val_json, tokenizer=tokenizer, data_mode=data_mode,
                                        augment=args.augment, num_tiles=args.num_tiles,
                                        negative_prompt_training=args.negative_prompt_training,
                                        block_embeddings=block_embeddings, batch_size=args.batch_size)

    train_dataset = train_dataloader.dataset

    first_sample = train_dataloader.dataset[0]
    scene_height = first_sample[0].shape[1]
    scene_width = first_sample[0].shape[2]

    print(f"Scene height: {scene_height}")
    print(f"Scene width: {scene_width}")

    if args.text_conditional:
        sample_captions, sample_negative_captions = gen_train_help.get_random_training_samples(train_dataloader, args.negative_prompt_training, args.output_dir)

    # if there is no block embedding model, set the channels to num_tiles
    in_channels = embedding_dim if args.block_embedding_model_path else args.num_tiles
    # else set channels to the embedding dimension of the model
    out_channels = in_channels


    # Setup the UNet model - use conditional version if text conditioning is enabled
    if args.text_conditional:
        model = UNet2DConditionModel(
            sample_size=(scene_height, scene_width),  # Fixed size for your level scenes
            in_channels=in_channels,  # Number of tile types (for one-hot encoding)
            out_channels=out_channels,
            layers_per_block=args.num_res_blocks,
            block_out_channels=[args.model_dim * mult for mult in args.dim_mults],
            down_block_types=args.down_block_types,
            up_block_types=args.up_block_types,
            cross_attention_dim=model_embedding_dim,  # Match the embedding dimension
            attention_head_dim=args.attention_head_dim,  # Number of attention heads
        )
        # Add flag for negative prompt support if enabled
        if args.negative_prompt_training:
            model.negative_prompt_support = True
    else:
        model = UNet2DModel(
            sample_size=(scene_height, scene_width),  # Fixed size for your level scenes
            in_channels=in_channels,  # Number of tile types (for one-hot encoding)
            out_channels=out_channels,
            layers_per_block=args.num_res_blocks,
            block_out_channels=[args.model_dim * mult for mult in args.dim_mults],
            down_block_types = [item.replace("CrossAttn", "") for item in args.down_block_types],
            up_block_types=[item.replace("CrossAttn", "") for item in args.up_block_types],
        )
    
    # Setup the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,  # Add weight decay to prevent overfitting
        betas=(0.9, 0.999)  # Default AdamW betas
    )
    
    # Setup learning rate scheduler
    total_training_steps = (len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_training_steps * args.lr_warmup_percentage)  

    print(f"Warmup period will be {warmup_steps} steps out of {total_training_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_cycles=args.lr_scheduler_cycles,
        num_warmup_steps=warmup_steps,  # Use calculated warmup steps
        num_training_steps=total_training_steps,
    )
    
    # Prepare for training with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
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
  
    # Add function to log metrics
    def log_metrics(epoch, loss, lr, step=None, val_loss=None):
        if accelerator.is_local_main_process:
            log_entry = {
                "epoch": epoch,
                "loss": loss,
                "lr": lr,
                "step": step if step is not None else epoch * len(train_dataloader),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if val_loss is not None:
                log_entry["val_loss"] = val_loss
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

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
    

    patience = args.patience if hasattr(args, 'patience') else 30
    best_val_loss = float('inf')
    best_caption_score = float('-inf')
    early_stop = False
    best_model_state = None
    # Track the epoch of the last improvement
    best_epoch = 0
    epochs_since_improvement = 0
    # If resuming training, load the latest checkpoint
    start_epoch = 0
    global_step = 0

    if resume_training:
        latest_ckpt, latest_epoch = find_latest_checkpoint(args.output_dir)
        # Handles log file(s) before resuming
        copy_log_up_to_epoch(args.output_dir, log_file, latest_epoch)
        if args.text_conditional and args.plot_validation_caption_score and caption_score_log_file:
            copy_log_up_to_epoch(args.output_dir, caption_score_log_file, latest_epoch)
        if latest_ckpt is not None:
            # Use pipeline's from_pretrained to load everything from the checkpoint directory
            if args.text_conditional:
                pipeline = TextConditionalDDPMPipeline.from_pretrained(latest_ckpt)
            else:
                pipeline = UnconditionalDDPMPipeline.from_pretrained(latest_ckpt)
            model = pipeline.unet
            noise_scheduler = pipeline.scheduler

            # Re-create the optimizer for the new model parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )

            # Load optimizer state if it exists
            optimizer_path = os.path.join(latest_ckpt, "optimizer.pt")
            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))

            # Load the learning rate scheduler state if it exists
            lr_scheduler_path = os.path.join(latest_ckpt, "lr_scheduler.pt")
            if os.path.exists(lr_scheduler_path):
                lr_scheduler.load_state_dict(torch.load(lr_scheduler_path, map_location="cpu"))

            # rewrap with accelerator
            model, optimizer = accelerator.prepare(model, optimizer)

            # After loading the pipeline and re-preparing with accelerator:
            early_stop_path = os.path.join(latest_ckpt, "early_stop_state.json")
            if os.path.exists(early_stop_path):
                with open(early_stop_path, "r") as f:
                    early_stop_state = json.load(f)
                best_val_loss = early_stop_state.get("best_val_loss", float('inf'))
                best_caption_score = early_stop_state.get("best_caption_score", float('-inf'))
                best_epoch = early_stop_state.get("best_epoch", 0)
                epochs_since_improvement = early_stop_state.get("epochs_since_improvement", 0)
            else:
                best_val_loss = float('inf')
                best_caption_score = float('-inf')
                best_epoch = 0
                epochs_since_improvement = 0
                
            start_epoch = latest_epoch + 1
            global_step = infer_global_step_from_log(log_file)
            print(f"Resumed training from epoch {start_epoch}, global_step {global_step}")
        else:
            print("Exiting from resumed training. No checkpoint found.") 
            exit()
            
    for epoch in range(start_epoch, args.num_epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss or caption score for {patience} epochs.")
            break
        model.train()
        train_loss = 0.0
        
        for batch in train_dataloader:
            # Add explicit memory clearing at start of batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with accelerator.accumulate(model):
                loss = process_diffusion_batch(
                    args, model, batch, noise_scheduler, loss_fn, tokenizer_hf, text_encoder, accelerator
                )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            train_loss += loss.detach().item()


            # Update progress bar
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
            
            # Detach tensors and clear memory
            del loss
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            
                        
            global_step += 1
        
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Calculate validation loss if validation dataset exists and it's time to validate
        val_loss = None
        avg_caption_score = None
        val_loss_improved = False
        caption_score_improved = False
        if val_dataloader is not None and (epoch % args.validate_epochs == 0 or epoch == args.num_epochs - 1):
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_batch_loss = process_diffusion_batch(
                        args, model, val_batch, noise_scheduler, loss_fn, tokenizer_hf, text_encoder, accelerator
                    )
                    val_loss += val_batch_loss.item()
                    # Clear memory after each validation batch
                    del val_batch_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            val_loss /= len(val_dataloader)

            if args.text_conditional and args.plot_validation_caption_score:
                # Compute caption match score for this data
                pipeline = TextConditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(model), 
                    scheduler=noise_scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer_hf if args.pretrained_language_model else None,
                    supports_pretrained_split=args.split_pretrained_sentences
                ).to(accelerator.device)
                # Only use the positive captions for scoring

                inference_steps = args.num_inference_timesteps
                # TODO: These should be argparse parameters
                guidance_scale = common_settings.GUIDANCE_SCALE
                avg_caption_score, _, _ = calculate_caption_score_and_samples(
                    accelerator.device, pipeline, val_dataloader, inference_steps, guidance_scale, args.seed,
                    id_to_char=id_to_char, char_to_id=char_to_id, tile_descriptors=tile_descriptors, describe_absence=args.describe_absence,
                    output=False, height=scene_height, width=scene_width
                )
            else:
                # Is this how this should behave in the unconditional case?
                # Or should I justs use 0 or -1?
                avg_caption_score = None

            model.train()

            # Log caption match score
            if args.text_conditional and args.plot_validation_caption_score and accelerator.is_local_main_process and caption_score_log_file:
                with open(caption_score_log_file, 'a') as f:
                    log_entry = {
                        "epoch": epoch,
                        "caption_score": avg_caption_score,                
                        "step": global_step,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    f.write(json.dumps(log_entry) + '\n')

            # Early stopping logic: check if EITHER metric improved in the epoch
            val_loss_improved = val_loss is not None and val_loss < best_val_loss
            caption_score_improved = avg_caption_score is not None and avg_caption_score > best_caption_score

            # Save best model if BOTH metrics improve, or if validation loss improves
            # CONSIDER: Save the model if either metric improves? Base improvement on the best of the two?
            if caption_score_improved:
                best_caption_score = avg_caption_score

            if val_loss_improved: # consider caption_score_improved too?
                best_val_loss = val_loss
                best_epoch = epoch

                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'caption_score': avg_caption_score,
                }

            # Early stopping logic: Conditional training end when both validation and caption metrics stop improving
            # and unconditional training ends when validation loss stops improving
            if args.text_conditional and args.plot_validation_caption_score:
                no_improvement = not val_loss_improved and not caption_score_improved
            else:
                no_improvement = not val_loss_improved

            if no_improvement:
                epochs_since_improvement = epoch - best_epoch
                if args.text_conditional and args.plot_validation_caption_score:
                    print(f"No improvement in val loss or caption score for {epochs_since_improvement}/{patience} epochs.")
                else:
                    print(f"No improvement in val loss for {epochs_since_improvement}/{patience} epochs.")
                if epochs_since_improvement >= patience:
                    if args.text_conditional and args.plot_validation_caption_score:
                        print(f"\nEarly stopping triggered. Best val loss: {best_val_loss:.4f}, Best caption score: {best_caption_score:.4f}")
                    else:
                        print(f"\nEarly stopping triggered. Best val loss: {best_val_loss:.4f}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state['model_state_dict'])
                    early_stop = True
        
        # Log metrics including validation loss
        log_metrics(epoch, avg_train_loss, lr_scheduler.get_last_lr()[0], val_loss=val_loss, step=global_step)
        
        # Print epoch summary (similar to train_mlm.py)
        if val_dataloader is not None and (epoch % args.validate_epochs == 0 or epoch == args.num_epochs - 1):
            val_result = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            caption_result = f"{avg_caption_score:.4f}" if avg_caption_score is not None else "N/A"
            print(
                f"Epoch {epoch+1} of {args.num_epochs}, "
                f"Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_result}, "
                f"Caption Score: {caption_result}, "
                f"No improvement in val loss or caption score for {epochs_since_improvement} of {patience} epochs."
            )
        else:
            print(
                f"Epoch {epoch+1} of {args.num_epochs}, "
                f"Loss: {avg_train_loss:.4f}, "
                f"No improvement in val loss for {epochs_since_improvement} of {patience} epochs."
            )

        # Generate and save sample levels every N epochs
        if epoch % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            # Switch to eval mode
            model.eval()
            
            # Create the appropriate pipeline for generation
            if args.text_conditional:
                pipeline = TextConditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(model), 
                    scheduler=noise_scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer_hf if args.pretrained_language_model else None, 
                    supports_pretrained_split=args.split_pretrained_sentences
                ).to(accelerator.device)
                                
                # Use the raw negative captions instead of tokens
                with torch.no_grad():
                    samples = pipeline(
                        batch_size=4,
                        generator=torch.Generator(device=accelerator.device).manual_seed(args.seed),
                        num_inference_steps = args.num_inference_timesteps, # Fewer steps needed for inference
                        output_type="tensor",
                        height=scene_height,
                        width=scene_width,
                        caption=sample_captions,
                        show_progress_bar=False,
                        negative_prompt=sample_negative_captions if args.negative_prompt_training else None 
                    ).images
            else:
                # For unconditional generation
                pipeline = UnconditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(model), 
                    scheduler=noise_scheduler
                )
                if sprite_scaling_factors is not None:
                    pipeline.give_sprite_scaling_factors(sprite_scaling_factors)

                
                # Generate sample levels
                with torch.no_grad():
                    samples = pipeline(
                        batch_size=4,
                        height=scene_height,
                        width=scene_width,
                        generator=torch.Generator(device=accelerator.device).manual_seed(args.seed),
                        num_inference_steps = args.num_inference_timesteps, # Fewer steps needed for inference
                        output_type="tensor",
                        show_progress_bar=False,
                    ).images

            # Convert one-hot samples to tile indices and visualize
            # TODO: Add prompt support
            prompts = sample_captions if args.text_conditional else None
            visualize_samples(samples, os.path.join(args.output_dir, f"samples_epoch_{epoch}"), prompts=prompts)
            
        # Save model every N epochs
        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}")
            # save the model
            if args.text_conditional:
                pipeline = TextConditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(model), 
                    scheduler=noise_scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer_hf if args.pretrained_language_model else None,
                    supports_pretrained_split=args.split_pretrained_sentences
                ).to(accelerator.device)
                # Save negative prompt support flag if enabled
                if args.negative_prompt_training:
                    pipeline.supports_negative_prompt = True
            else:
                pipeline = UnconditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(model), 
                    scheduler=noise_scheduler
                )
                if sprite_scaling_factors is not None:
                    pipeline.give_sprite_scaling_factors(sprite_scaling_factors)
            # Wait for all processes to synchronize before saving
            accelerator.wait_for_everyone()
            pipeline.save_pretrained(checkpoint_dir)
            # Save optimizer state
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            # Save the optimizer state dictionary
            torch.save(optimizer.state_dict(), optimizer_path)
            # Save LR scheduler state
            lr_scheduler_path = os.path.join(checkpoint_dir, "lr_scheduler.pt")
            torch.save(lr_scheduler.state_dict(), lr_scheduler_path)

            # Save early stopping state
            early_stop_state = {
                "best_val_loss": best_val_loss,
                "best_caption_score": best_caption_score,
                "best_epoch": best_epoch,
                "epochs_since_improvement": epochs_since_improvement
            }
            early_stop_path = os.path.join(checkpoint_dir, "early_stop_state.json")
            with open(early_stop_path, "w") as f:
                json.dump(early_stop_state, f)
            
    try:
        # Clean up plotting resources
        if accelerator.is_local_main_process and plotter:
            # Better thread cleanup
            gen_train_help.kill_plotter(plotter, plot_thread)

            gen_train_help.kill_plotter(caption_score_plotter, caption_score_plot_thread)

        # Force CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Ensure all processes are synchronized
        accelerator.wait_for_everyone()

    finally:
        # Close progress bar and TensorBoard writer
        progress_bar.close()
        
        # Final model save
        if args.text_conditional:
            pipeline = TextConditionalDDPMPipeline(
                unet=accelerator.unwrap_model(model), 
                scheduler=noise_scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer_hf if args.pretrained_language_model else None,
                supports_pretrained_split=args.split_pretrained_sentences
            ).to(accelerator.device)
        else:
            pipeline = UnconditionalDDPMPipeline(
                unet=accelerator.unwrap_model(model), 
                scheduler=noise_scheduler
            )
            if sprite_scaling_factors is not None:
                pipeline.give_sprite_scaling_factors(sprite_scaling_factors)
            
        pipeline.save_pretrained(args.output_dir)
        # # Save the final optimizer and learing rate scheduler states??
        # optimizer_path = os.path.join(args.output_dir, "optimizer.pt")
        # torch.save(optimizer.state_dict(), optimizer_path)
        # lr_scheduler_path = os.path.join(args.output_dir, "lr_scheduler.pt")
        # torch.save(lr_scheduler.state_dict(), lr_scheduler_path)
        
# Add function to load config from JSON
def load_config_from_json(config_path):
    """Load hyperparameters from a JSON config file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Configuration loaded from {config_path}")
            
            # Print the loaded config for verification
            print("Loaded hyperparameters:")
            for key, value in config.items():
                print(f"  {key}: {value}")
                
            return config
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading config file: {e}")
        raise e

def update_args_from_config(args, config):
    """Update argparse namespace with values from config."""
    # Convert config dict to argparse namespace
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args

def prepare_conditioned_batch(args, tokenizer_hf, text_encoder, scenes, captions, timesteps, device, negative_captions=None):
    """
    Prepares the batch for training with text conditioning.

    Embedding shape expectations:
    - If args.split_pretrained_sentences: 
        combined_embeddings shape is [batch, num_phrases, embedding_dim]
    - If args.pretrained_language_model (no split): 
        combined_embeddings shape is [batch, 1, embedding_dim]
    - Else (token embedding): 
        combined_embeddings shape is [batch, num_tokens, embedding_dim]

    Returns:
        combined_embeddings: torch.Tensor
        scenes_for_train: torch.Tensor
        timesteps_for_train: torch.Tensor
    """
    #Prepares the batch for training with text conditioning.
    with torch.no_grad():         
        if args.split_pretrained_sentences:
            # Each caption is split into phrases; embedding shape: [batch, num_phrases, embedding_dim]
            combined_embeddings = st_helper.get_embeddings_split(batch_size=len(captions),
                                                       tokenizer=tokenizer_hf,
                                                       model=text_encoder,
                                                       captions=captions,
                                                       neg_captions=negative_captions,
                                                       device=device)
        elif args.pretrained_language_model:
            # Each caption is embedded as a single vector; shape: [batch, 1, embedding_dim]
            combined_embeddings = st_helper.get_embeddings(batch_size=len(captions),
                                                       tokenizer=tokenizer_hf,
                                                       model=text_encoder,
                                                       captions=captions,
                                                       neg_captions=negative_captions,
                                                       device=device)
            
        else:
            # Token-level embedding; shape: [batch, num_tokens, embedding_dim]
            combined_embeddings = text_model.get_embeddings(batch_size=len(captions),
                                                       tokenizer=text_encoder.tokenizer,
                                                       text_encoder=text_encoder,
                                                       captions=captions,
                                                       neg_captions=negative_captions,
                                                       device=device)

        repeat_factor = 3 if args.negative_prompt_training else 2
        if args.split_pretrained_sentences:
            # [batch, num_phrases, embedding_dim]
            assert combined_embeddings.ndim == 3, "Expected [batch, num_phrases, embedding_dim] for split_pretrained_sentences"
            assert combined_embeddings.shape[0] == len(captions)*repeat_factor, f"Batch size mismatch in split_pretrained_sentences: shape {combined_embeddings.shape} and captions {len(captions)}"
        elif args.pretrained_language_model:
            # [batch, 1, embedding_dim]
            assert combined_embeddings.ndim == 3, "Expected [batch, 1, embedding_dim] for pretrained_language_model"
            assert combined_embeddings.shape[0] == len(captions)*repeat_factor, f"Batch size mismatch in pretrained_language_model: shape {combined_embeddings.shape} and captions {len(captions)}"
            assert combined_embeddings.shape[1] == 1, f"Expected singleton phrase dimension for pretrained_language_model: shape {combined_embeddings.shape}"
        else:
            # [batch, num_tokens, embedding_dim]
            assert combined_embeddings.ndim == 3, "Expected [batch, num_tokens, embedding_dim] for token embedding"
            assert combined_embeddings.shape[0] == len(captions)*repeat_factor, f"Batch size mismatch in token embedding: shape {combined_embeddings.shape} and captions {len(captions)}"

        if args.negative_prompt_training:
            scenes_for_train = torch.cat([scenes] * 3)  # Repeat scenes three times
            timesteps_for_train = torch.cat([timesteps] * 3)  # Repeat timesteps three times
        else:
            # Original classifier-free guidance with just uncond and cond
            scenes_for_train = torch.cat([scenes] * 2)  # Repeat scenes twice
            timesteps_for_train = torch.cat([timesteps] * 2)  # Repeat timesteps twice

        return combined_embeddings, scenes_for_train, timesteps_for_train

def process_diffusion_batch(
    args, model, batch, noise_scheduler, loss_fn, tokenizer_hf, text_encoder, accelerator
):
    """
    Handles a single batch for training or validation.
    """ 
    if args.negative_prompt_training:
        scenes, captions, negative_captions = batch
    else:
        scenes, captions = batch
        negative_captions = None

    scenes = scenes.to(accelerator.device)

    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (scenes.shape[0],), device=accelerator.device
    ).long()
    

    if args.text_conditional: #Here's the big difference between the two training modes
        #If we're using text conditioning, we need to prepare the embeddings
        combined_embeddings, scenes_for_train, timesteps_for_train = prepare_conditioned_batch(
            args, tokenizer_hf, text_encoder, scenes, captions, timesteps, accelerator.device, negative_captions=negative_captions
        )
    else: #Otherwise they can be set as is
        combined_embeddings, scenes_for_train, timesteps_for_train = None, scenes, timesteps

    noise = torch.randn_like(scenes_for_train)
    noisy_scenes = noise_scheduler.add_noise(scenes_for_train, noise, timesteps_for_train)
    
    if args.text_conditional:
        noise_pred = model(noisy_scenes, timesteps_for_train, encoder_hidden_states=combined_embeddings).sample
    else: # unconditional model does not allow encoder_hidden_states parameter
        noise_pred = model(noisy_scenes, timesteps_for_train).sample

    target_noise = noise
    batch_loss = loss_fn(
        noise_pred, target_noise, scenes_for_train, noisy_scenes,
        timesteps=timesteps_for_train, scheduler=noise_scheduler
    )
    return batch_loss

if __name__ == "__main__":
     main()