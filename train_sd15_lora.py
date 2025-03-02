# From https://github.com/PacktPublishing/Using-Stable-Diffusion-with-Python/blob/main/chapter_21/train_sd15_lora.py
# Modified with checkpointing, logging, real-time plotting, and reproducibility features

# import packages
import torch
from accelerate import utils
from accelerate import Accelerator
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from datasets import load_dataset
from torchvision import transforms
import math
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers.utils import convert_state_dict_to_diffusers
import argparse
import os
import json
from datetime import datetime

# For plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import matplotlib
import sys

formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')

class LossPlotter:
    def __init__(self, log_file, update_interval=1.0, interactive=False):
        self.log_file = log_file
        self.update_interval = update_interval
        self.interactive = interactive
        
        # Set the backend based on whether we're in interactive mode
        if interactive:
            # Try to use an interactive backend if available
            try:
                # Check if running in a GUI-capable environment
                if sys.platform.startswith('win'):
                    matplotlib.use('TkAgg')
                elif sys.platform.startswith('darwin'):  # macOS
                    matplotlib.use('MacOSX')
                else:  # Linux and others
                    # Try Qt first, fall back to TkAgg
                    try:
                        matplotlib.use('Qt5Agg')
                    except ImportError:
                        matplotlib.use('TkAgg')
            except Exception as e:
                print(f"Warning: Could not set interactive backend ({e}). Falling back to non-interactive mode.")
                self.interactive = False
                matplotlib.use('Agg')
        else:
            # Use non-interactive backend
            matplotlib.use('Agg')
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.epochs = []
        self.losses = []
        self.lr_values = []
        self.running = True
        self.ani = None
        
    def update_plot(self, frame):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = [json.loads(line) for line in f if line.strip()]
                    
                if not data:
                    return self.ax,
                    
                self.epochs = [entry.get('epoch', 0) for entry in data]
                self.losses = [entry.get('loss', 0) for entry in data]
                self.lr_values = [entry.get('lr', 0) for entry in data]
                
                # Clear the axes and redraw
                self.ax.clear()
                # Plot loss
                loss_line, = self.ax.plot(self.epochs, self.losses, 'b-', label='Training Loss')
                self.ax.set_xlabel('Epoch')
                self.ax.set_ylabel('Loss', color='b')
                self.ax.tick_params(axis='y', labelcolor='b')
                
                # Add learning rate on secondary y-axis if available
                if any(self.lr_values):
                    ax2 = self.ax.twinx()
                    lr_line, = ax2.plot(self.epochs, self.lr_values, 'r-', label='Learning Rate')
                    ax2.set_ylabel('Learning Rate', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                
                # Add a title and legend
                self.ax.set_title('Training Progress')
                self.ax.legend(loc='upper left')
                if any(self.lr_values):
                    ax2.legend(loc='upper right')
                
                # Adjust layout
                self.fig.tight_layout()
                
                # Save the current plot to disk
                self.fig.savefig(os.path.join(os.path.dirname(self.log_file), 'current_progress.png'))
            
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing log file: {e}")
        
        return self.ax,
    
    def start_plotting(self):
        if self.interactive:
            self.ani = FuncAnimation(
                self.fig, self.update_plot, interval=self.update_interval * 1000, cache_frame_data=False
            )
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not show interactive plot ({e}). Progress will still be saved as images.")
                # Continue without showing the plot
        else:
            # For non-interactive mode, periodically update the saved image
            while self.running:
                self.update_plot(0)
                time.sleep(self.update_interval)
    
    def stop_plotting(self):
        self.running = False
        if self.ani and self.interactive:
            self.ani.event_source.stop()
        plt.close(self.fig)

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
    
    # Handle special cases for nested parameters
    if 'adam_betas' in config and isinstance(config['adam_betas'], list) and len(config['adam_betas']) == 2:
        args.adam_betas = config['adam_betas']
    
    return args

def generate_samples(pipe, epoch, output_dir, prefix, prompt, resolution, num_samples=4, guidance_scale=7.5, steps=30):
    """Generate and save sample images to track model progress."""
    # Create samples directory if it doesn't exist
    samples_dir = os.path.join(output_dir, f"{prefix}_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Set evaluation mode
    pipe.unet.eval()
    
    # Generate samples with current model state
    with torch.no_grad():
        # Generate multiple samples at once
        images = pipe([prompt] * num_samples, 
                      num_inference_steps=steps,
                      height = resolution,
                      width = resolution,
                      guidance_scale=guidance_scale).images
        
        # Save each image
        for i, image in enumerate(images):
            image_path = os.path.join(samples_dir, f"epoch_{epoch:03d}_sample_{i+1}.png")
            image.save(image_path)
            
    print(f"Generated {num_samples} sample images for epoch {epoch}")
    
    # Return to training mode
    pipe.unet.train()
    
    return samples_dir

# train code 
def main(args):
    # If config file is provided, load and update arguments
    if args.config:
        config = load_config_from_json(args.config)
        args = update_args_from_config(args, config)
        print("Training will use parameters from the config file, overridden by any explicitly provided command line arguments.")

    utils.write_basic_config()

    # hyperparameters from command line arguments with defaults
    pretrained_model_name_or_path = args.pretrained_model
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    learning_rate = args.learning_rate
    adam_beta1, adam_beta2 = args.adam_betas
    adam_weight_decay = args.adam_weight_decay
    adam_epsilon = args.adam_epsilon
    dataset_name = args.dataset_name
    train_data_dir = args.train
    top_rows = args.top_rows
    output_dir = args.output
    resolution = args.resolution
    center_crop = args.center_crop
    random_flip = args.random_flip
    train_batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_epochs = args.epochs
    lr_scheduler_name = args.lr_scheduler
    max_grad_norm = args.max_grad_norm
    diffusion_scheduler = DDPMScheduler
    prefix = args.save  # Save name
    checkpoint_interval = args.checkpoint_interval

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file for loss tracking
    log_file = os.path.join(output_dir, f"{prefix}_training_log_{formatted_date}.jsonl")
    
    # Setup real-time plotting if requested
    plotter = None
    plot_thread = None
    if args.plot_loss:
        # Determine if we should try to use interactive mode
        try_interactive = args.interactive_plot
        plotter = LossPlotter(log_file, interactive=try_interactive)
        plot_thread = threading.Thread(target=plotter.start_plotting)
        plot_thread.daemon = True
        plot_thread.start()
        print(f"Loss plotting enabled: {'interactive' if try_interactive else 'non-interactive'} mode")
        print(f"Progress images will be saved to {output_dir}/current_progress.png")

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16"
    )
    device = accelerator.device

    # Log the chosen hyperparameters
    hyperparams = {
        "pretrained_model": pretrained_model_name_or_path,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "adam_betas": [adam_beta1, adam_beta2],
        "adam_weight_decay": adam_weight_decay,
        "adam_epsilon": adam_epsilon,
        "dataset_name": dataset_name,
        "train_data_dir": train_data_dir,
        "top_rows": top_rows,
        "resolution": resolution,
        "center_crop": center_crop,
        "random_flip": random_flip,
        "batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epochs": num_train_epochs,
        "lr_scheduler": lr_scheduler_name,
        "max_grad_norm": max_grad_norm,
        "checkpoint_interval": checkpoint_interval,
        "warmup_ratio": args.warmup_ratio,
        "date": formatted_date
    }
    
    # Save hyperparameters to a JSON file
    config_filename = f"{prefix}_hyperparams_{formatted_date}.json"
    config_path = os.path.join(output_dir, config_filename)
    with open(config_path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    print(f"Saved configuration file to: {config_path}")
    print(f"To reproduce this training run, use: python {os.path.basename(__file__)} --config {config_path}")
    
    # Load scheduler, tokenizer and unet models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    weight_dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype=weight_dtype
    ).to(device)
    tokenizer, text_encoder, vae, unet = pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet

    # freeze parameters of models, we just want to train a LoRA only
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # configure LoRA parameters use PEFT
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    for param in unet.parameters():
        # only upcast trainable parameters (LoRA) into fp32
        if param.requires_grad:
            param.data = param.to(torch.float32)
    
    # Downloading and loading a dataset from the hub. data will be saved to ~/.cache/huggingface/datasets by default
    if dataset_name:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(
            "imagefolder",
            data_dir=train_data_dir
        )
    
    train_data = dataset["train"]
    if top_rows > 0:
        dataset["train"] = train_data.select(range(min(top_rows, len(train_data))))
    print(dataset["train"])

    # Preprocessing the datasets. We need to tokenize inputs and targets.
    dataset_columns = list(dataset["train"].features.keys())
    image_column, caption_column = dataset_columns[0], dataset_columns[1]

    def tokenize_captions(examples, is_train=True):
        '''Preprocessing the datasets.We need to tokenize input captions and transform the images.'''
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
        ]
    )

    def preprocess_train(examples):
        '''prepare the train data'''
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    # only do this in the main process
    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=0
    )

    print("Data Size:", len(train_dataloader))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_batch_size)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # initialize optimizer
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon
    )

    # learn rate scheduler from diffusers's get_scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * max_train_steps),
        num_training_steps=max_train_steps
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # set step count and progress bar
    max_train_steps = num_train_epochs * len(train_dataloader)
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Function to save checkpoints
    def save_checkpoint(unwrapped_unet, epoch):
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        checkpoint_name = f"{prefix}_lora_{pretrained_model_name_or_path.split('/')[-1]}_rank{lora_rank}_epoch{epoch}_r{resolution}_{diffusion_scheduler.__name__}_{formatted_date}.safetensors"   
        StableDiffusionPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
            weight_name=checkpoint_name
        )
        print(f"Checkpoint saved at epoch {epoch}: {checkpoint_name}")

    # Function to log training metrics
    def log_metrics(epoch, loss, lr, step=None):
        log_entry = {
            "epoch": epoch,
            "loss": loss,
            "lr": lr,
            "step": step if step is not None else epoch * len(train_dataloader),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    # start train
    epoch_losses = []
    for epoch in range(num_train_epochs):
        unet.train()
        train_loss = 0.0
        step_losses = []
        
        for step, batch in enumerate(train_dataloader):
            # step 1. Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # step 2. Sample noise that we'll add to the latents, latents provide the shape info. 
            noise = torch.randn_like(latents)

            # step 3. Sample a random timestep for each image
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                low=0,
                high=noise_scheduler.config.num_train_timesteps,
                size=(batch_size,),
                device=latents.device
            )
            timesteps = timesteps.long()

            # step 4. Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # step 5. Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process), provide to unet to get the prediction result
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # step 6. Get the target for loss depend on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # step 7. Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # step 8. Calculate loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # step 9. Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps
            step_losses.append(avg_loss.item())

            # step 10. Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = lora_layers
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step 11. check optimization step and update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                
                # Log step-level information if needed
                if step % args.log_steps == 0:
                    log_metrics(epoch, avg_loss.item(), lr_scheduler.get_last_lr()[0], step=step)
                
                train_loss = 0.0
            
            logs = {"epoch": epoch, "step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
        
        # Calculate and log epoch average loss
        epoch_avg_loss = sum(step_losses) / len(step_losses) if step_losses else 0
        epoch_losses.append(epoch_avg_loss)
        log_metrics(epoch, epoch_avg_loss, lr_scheduler.get_last_lr()[0])
        
        print(f"Epoch {epoch}: Average Loss = {epoch_avg_loss:.6f}")
        
        # Save checkpoint at specified intervals
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet = unwrapped_unet.to(torch.float32)
                save_checkpoint(unwrapped_unet, epoch + 1)

        # Generate sample images at specified intervals
        if args.sample_interval > 0 and (epoch + 1) % args.sample_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # Temporarily convert model to float32
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet = unwrapped_unet.to(torch.float32)
        
                # Create a pipeline with the current state of the model
                eval_pipe = StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path,
                    unet=unwrapped_unet,
                    torch_dtype=weight_dtype
                ).to(device)
        
                # Generate and save samples
                samples_dir = generate_samples(
                    eval_pipe, 
                    epoch + 1, 
                    output_dir, 
                    prefix, 
                    args.sample_prompt,
                    resolution=args.resolution,
                    num_samples=args.num_samples
                )
        
        # Log the sample generation
        log_entry = {
            "epoch": epoch + 1,
            "samples_dir": samples_dir,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(output_dir, f"{prefix}_samples_log.json"), 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Restore unet to its previous state
        unwrapped_unet = unwrapped_unet.to(weight_dtype)

    # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

        # Save the final model
        weight_name = f"{prefix}_lora_{pretrained_model_name_or_path.split('/')[-1]}_rank{lora_rank}_s{max_train_steps}_r{resolution}_{diffusion_scheduler.__name__}_{formatted_date}.safetensors"   
        StableDiffusionPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
            weight_name=weight_name
        )
        
        # Save a final plot of the training loss
        if epoch_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(range(num_train_epochs), epoch_losses, 'b-', label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_loss_plot_{formatted_date}.png"))
            plt.close()

    # Stop the plotter if it's running
    if plotter:
        plotter.stop_plotting()
        if plot_thread and plot_thread.is_alive():
            plot_thread.join(timeout=1.0)

    accelerator.end_training()

def generate_config_template():
    """Generate a template config file with all available parameters and their defaults."""
    # Create a parser just to get the default values
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])  # Parse empty args to get defaults
    
    # Convert args to dictionary
    config = vars(args)
    
    # Remove the config parameter itself
    if 'config' in config:
        del config['config']
    
    # Handle special cases like adam_betas which is a tuple
    if hasattr(args, 'adam_betas'):
        config['adam_betas'] = list(args.adam_betas)
    
    # Generate template file
    template_path = "lora_config_template.json"
    with open(template_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return template_path

def add_arguments(parser):
    """Add all arguments to the parser - separated to be reusable."""
    # Core parameters
    parser.add_argument("-t", "--train", type=str, default="train_data", help="Directory to get training data from.")
    parser.add_argument("-o", "--output", type=str, default="output_dir", help="Directory to output model to.")
    parser.add_argument("-r", "--resolution", type=int, default=768, help="Convert training images to this resolution (square).")
    parser.add_argument("-s", "--save", type=str, default="", help="Prefix of saved model name.")
    
    # Config loading
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file with training parameters.")
    parser.add_argument("--generate_config", action="store_true", help="Generate a template config file and exit.")
    
    # Model configuration
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Pretrained model name or path")
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank for LoRA training")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha for LoRA training")
    
    # Dataset options
    parser.add_argument("--dataset_name", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--top_rows", type=int, default=0, help="Number of rows to use from dataset (0 means all of them)")
    parser.add_argument("--center_crop", action="store_true", help="Use center crop instead of random crop")
    parser.add_argument("--random_flip", action="store_true", help="Sometimes flip training images")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.999], help="Adam betas")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay for Adam")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Type of learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    
    # Checkpointing and logging
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Save checkpoints every N epochs (0 to disable)")
    parser.add_argument("--log_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--plot_loss", action="store_true", help="Plot loss during training")
    parser.add_argument("--interactive_plot", action="store_true", help="Try to use an interactive plot window (if supported)")

    # Image samples
    parser.add_argument("--sample_interval", type=int, default=10, help="Generate sample images every N epochs (0 to disable)")
    parser.add_argument("--sample_prompt", type=str, default="overworld level. blue sky. full floor. several bricks. a few questionblocks in a horizontal line. many solidblocks. three goombas. a brickledge. a cloud. a obstacle. a greenpipe. stairs", help="Prompt to use for sample image generation")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of sample images to generate")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a LoRA for Stable Diffusion.")
    add_arguments(parser)
    args = parser.parse_args()
    
    # Check if user wants to generate a template config
    if args.generate_config:
        template_path = generate_config_template()
        print(f"Generated template config file at: {template_path}")
        print("You can modify this file and use it with --config option.")
        exit(0)

    main(args)
