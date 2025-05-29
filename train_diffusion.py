import argparse
import os
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup 
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
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
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
        config = load_config_from_json(args.config)
        args = update_args_from_config(args, config)
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

    # Initialize dataset
    train_dataset = LevelDataset(
        json_path=args.json,
        tokenizer=tokenizer,
        shuffle=True,
        mode=data_mode,
        augment=args.augment,
        num_tiles=args.num_tiles,
        negative_captions=args.negative_prompt_training,
        block_embeddings=block_embeddings
    )
    val_dataset = None
    if args.val_json is not None:
        val_dataset = LevelDataset(
            json_path=args.val_json,
            tokenizer=tokenizer,
            shuffle=False,
            mode=data_mode,
            augment=False,
            num_tiles=args.num_tiles,
            negative_captions=args.negative_prompt_training,
            block_embeddings=block_embeddings
        )

    first_sample = train_dataset[0]
    scene_height = first_sample[0].shape[1]
    scene_width = first_sample[0].shape[2]

    print(f"Scene height: {scene_height}")
    print(f"Scene width: {scene_width}")

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )

    if args.text_conditional:
        # Sample four random captions from the dataset
        sample_indices = [random.randint(0, len(train_dataset) - 1) for _ in range(4)]

        sample_captions = [train_dataset[i][1] for i in sample_indices]
        print("Sample captions:")
        for caption in sample_captions:
            print(caption)

        if args.negative_prompt_training:
            sample_negative_captions = [train_dataset[i][2] for i in sample_indices]
            print("Sample negative captions:")
            for caption in sample_negative_captions:
                print(f"  NEG: {caption}")

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
    
    # # Second occurance to create new directory. Delete?
    # # Create output directory
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # else:
    #     print(f"Output directory '{args.output_dir}' already exists. Please remove it or choose a different name.")
    #     exit()
    
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
    plotter = None
    plot_thread = None
    caption_score_plotter = None
    caption_score_plot_thread = None
    caption_score_log_file = None
    if accelerator.is_local_main_process:
        plotter = Plotter(log_file, update_interval=5.0, left_key='loss', right_key='val_loss',
                             left_label='Training Loss', right_label='Validation Loss', output_png=f'training_loss_{formatted_date}.png')
        plot_thread = threading.Thread(target=plotter.start_plotting)
        plot_thread.daemon = True
        plot_thread.start()
        print(f"Loss plotting enabled. Progress will be saved to {os.path.join(args.output_dir, f'training_loss_{formatted_date}.png')}")
        caption_score_plotter = None
        if args.text_conditional and args.plot_validation_caption_score:
            caption_score_log_file = os.path.join(args.output_dir, f"caption_score_log_{formatted_date}.jsonl")
            # Caption score plotter
            caption_score_plotter = Plotter(caption_score_log_file, update_interval=5.0, left_key='caption_score', right_key=None,
                                                left_label='Caption Match Score', right_label=None, output_png=f'caption_score_{formatted_date}.png')
            caption_score_plot_thread = threading.Thread(target=caption_score_plotter.start_plotting)
            caption_score_plot_thread.daemon = True
            caption_score_plot_thread.start()
            print(f"Caption match score plotting enabled. Progress will be saved to {os.path.join(args.output_dir, f'caption_score_{formatted_date}.png')}")

            _, id_to_char, char_to_id, tile_descriptors = extract_tileset(args.tileset)

    best_val_loss = float('inf')
    best_caption_score = float('-inf')
    best_model_state = None
    # Track the epoch of the last improvement
    
    # Initialize variables to track the best model 
    best_val_loss = float('inf')
    best_caption_score = float('-inf')
    best_model_state = None

    for epoch in range(args.num_epochs):
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
                avg_caption_score, _ = calculate_caption_score_and_samples(
                    accelerator.device, pipeline, val_dataloader, inference_steps, guidance_scale, args.seed,
                    id_to_char=id_to_char, char_to_id=char_to_id, tile_descriptors=tile_descriptors, describe_absence=args.describe_absence,
                    output=False, height=scene_height, width=scene_width
                )
            else:
                # Is this how this should behave in the unconditional case?
                # Or should I justs use 0 or -1?
                avg_caption_score = None

            model.train()
            
            # Update the best model based on the chosen criterion
            if args.best_model_criterion == "val_loss" and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'caption_score': avg_caption_score,
                }
            elif args.best_model_criterion == "caption_score" and avg_caption_score is not None and avg_caption_score > best_caption_score:
                best_caption_score = avg_caption_score
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'caption_score': avg_caption_score,
                }

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

            # # Early stopping logic: Conditional training end when both validation and caption metrics stop improving
            # # and unconditional training ends when validation loss stops improving
            # if args.text_conditional and args.plot_validation_caption_score:
            #     no_improvement = not val_loss_improved and not caption_score_improved
            # else:
            #     no_improvement = not val_loss_improved

            # if no_improvement:
            #     epochs_since_improvement = epoch - best_epoch
            #     if args.text_conditional and args.plot_validation_caption_score:
            #         print(f"No improvement in val loss or caption score for {epochs_since_improvement}/{patience} epochs.")
            #     else:
            #         print(f"No improvement in val loss for {epochs_since_improvement}/{patience} epochs.")
            #     if epochs_since_improvement >= patience:
            #         if args.text_conditional and args.plot_validation_caption_score:
            #             print(f"\nEarly stopping triggered. Best val loss: {best_val_loss:.4f}, Best caption score: {best_caption_score:.4f}")
            #         else:
            #             print(f"\nEarly stopping triggered. Best val loss: {best_val_loss:.4f}")
            #         if best_model_state is not None:
            #             model.load_state_dict(best_model_state['model_state_dict'])
            #         early_stop = True
        
        # Log metrics including validation loss
        log_metrics(epoch, avg_train_loss, lr_scheduler.get_last_lr()[0], val_loss=val_loss, step=global_step)
        
        # Print epoch summary (similar to train_mlm.py)
        if val_dataloader is not None and (epoch % args.validate_epochs == 0 or epoch == args.num_epochs - 1):
            print(
                f"Epoch {epoch+1}/{args.num_epochs}, "
                f"Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f if val_loss is not None else 'N/A'}, "
                f"Caption Score: {avg_caption_score if avg_caption_score is not None else 'N/A'}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{args.num_epochs}, "
                f"Loss: {avg_train_loss:.4f}"
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
            # Save the model
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
                
            # Ensure all processes are synchronized
            accelerator.wait_for_everyone()
            pipeline.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch}"))
            
        # Save the best model at the end of training
        if best_model_state is not None:
            print(f"Saving the best model from epoch {best_model_state['epoch'] + 1}")
            model.load_state_dict(best_model_state['model_state_dict'])
            pipeline = TextConditionalDDPMPipeline(
                unet=accelerator.unwrap_model(model), 
                scheduler=noise_scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer_hf if args.pretrained_language_model else None
            ).to(accelerator.device)
            pipeline.save_pretrained(os.path.join(args.output_dir, "best_model"))
    
    try:
        # Clean up plotting resources
        if accelerator.is_local_main_process and plotter:
            # Better thread cleanup
            if plot_thread and plot_thread.is_alive():
                plotter.stop_plotting()
                plot_thread.join(timeout=5.0)
                if plot_thread.is_alive():
                    print("Warning: Plot thread did not terminate properly")
            if caption_score_plot_thread and caption_score_plot_thread.is_alive():
                caption_score_plotter.stop_plotting()
                caption_score_plot_thread.join(timeout=5.0)
                if caption_score_plot_thread.is_alive():
                    print("Warning: Caption score plot thread did not terminate properly")

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
    #Prepares the batch for training with text conditioning.
    with torch.no_grad():         
        if args.split_pretrained_sentences:
            combined_embeddings = st_helper.get_embeddings_split(batch_size=len(captions),
                                                       tokenizer=tokenizer_hf,
                                                       model=text_encoder,
                                                       captions=captions,
                                                       neg_captions=negative_captions,
                                                       device=device)
        elif args.pretrained_language_model:
            combined_embeddings = st_helper.get_embeddings(batch_size=len(captions),
                                                       tokenizer=tokenizer_hf,
                                                       model=text_encoder,
                                                       captions=captions,
                                                       neg_captions=negative_captions,
                                                       device=device)
            
        else:
            combined_embeddings = text_model.get_embeddings(batch_size=len(captions),
                                                       tokenizer=text_encoder.tokenizer,
                                                       text_encoder=text_encoder,
                                                       captions=captions,
                                                       neg_captions=negative_captions,
                                                       device=device)

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
    
    noise_pred = model(noisy_scenes, timesteps_for_train, encoder_hidden_states=combined_embeddings).sample

    target_noise = noise
    batch_loss = loss_fn(
        noise_pred, target_noise, scenes_for_train, noisy_scenes,
        timesteps=timesteps_for_train, scheduler=noise_scheduler
    )
    return batch_loss

if __name__ == "__main__":
    main()
