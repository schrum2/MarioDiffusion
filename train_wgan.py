import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random
import numpy as np
from level_dataset import LevelDataset, visualize_samples
import json
import threading
from datetime import datetime
from util.plotter import Plotter
from tokenizer import Tokenizer
from models.wgan_model import WGAN_Generator, WGAN_Discriminator

# For learning rate scheduler
from torch.optim.lr_scheduler import LambdaLR

def parse_args():
    parser = argparse.ArgumentParser(description="Train a WGAN for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
            
    # WGAN specific args
    parser.add_argument("--nz", type=int, default=32, help="Size of the latent z vector")
    parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=64, help="Size of feature maps in discriminator")
    parser.add_argument("--clamp_lower", type=float, default=-0.01, help="Lower bound for WGAN weight clamping")
    parser.add_argument("--clamp_upper", type=float, default=0.01, help="Upper bound for WGAN weight clamping")
    parser.add_argument("--n_critic", type=int, default=5, help="Number of discriminator iterations per generator iteration")
    parser.add_argument("--n_extra_layers", type=int, default=0, help="Number of extra layers in generator and discriminator")
    
    # Training args
    parser.add_argument("--lr_g", type=float, default=0.00005, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=0.00005, help="Discriminator learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--use_rmsprop", action="store_true", help="Use RMSprop optimizer instead of Adam")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--save_image_epochs", type=int, default=100, help="Save generated levels every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=100, help="Save model every N epochs")
    
    # Learning rate scheduling (optional)
    parser.add_argument("--use_lr_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--lr_warmup_percentage", type=float, default=0.05, help="Learning rate warmup portion")
    parser.add_argument("--lr_scheduler_cycles", type=float, default=0.5, help="Number of cycles for cosine learning rate scheduler")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="wgan-output", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    
    # Optional config file
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file with training parameters")

    return parser.parse_args()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Creates a learning rate scheduler with linear warmup and cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

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

def main():
    args = parse_args()

    # Check if config file is provided
    if args.config:
        config = load_config_from_json(args.config)
        args = update_args_from_config(args, config)
        print("Training will use parameters from the config file.")
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)

    # Initialize dataset
    dataset = LevelDataset(
        json_path=args.json,
        tokenizer=tokenizer,
        shuffle=True,
        mode="diffusion",  # We need this for compatibility with your dataset class
        augment=args.augment,
        num_tiles=args.num_tiles
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # Set input image size (16x16 for your level data)
    isize = 16
    
    # Initialize generator and discriminator
    netG = WGAN_Generator(isize, args.nz, args.num_tiles, args.ngf, n_extra_layers=args.n_extra_layers)
    netD = WGAN_Discriminator(isize, args.num_tiles, args.ndf, n_extra_layers=args.n_extra_layers)
    
    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Move models to device
    netG = netG.to(device)
    netD = netD.to(device)
    
    print(netG)
    print(netD)
    
    # Set up fixed noise for testing
    fixed_noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1).normal_(0, 1).to(device)
    
    # Set up optimizers
    if args.use_rmsprop:
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr_g)
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lr_d)
        print("Using RMSprop optimizer")
    else:
        optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))
        print("Using Adam optimizer")
    
    # Set up learning rate schedulers if requested
    lr_schedulerG = None
    lr_schedulerD = None
    
    if args.use_lr_scheduler:
        total_training_steps = len(dataloader) * args.num_epochs
        warmup_steps = int(total_training_steps * args.lr_warmup_percentage)
        
        print(f"Using learning rate scheduler with warmup: {warmup_steps} steps out of {total_training_steps}")
        
        lr_schedulerG = get_cosine_schedule_with_warmup(
            optimizer=optimizerG,
            num_cycles=args.lr_scheduler_cycles,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
        
        lr_schedulerD = get_cosine_schedule_with_warmup(
            optimizer=optimizerD,
            num_cycles=args.lr_scheduler_cycles,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
    
    # Get formatted timestamp for filenames
    formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')

    # Create log files
    log_file = os.path.join(args.output_dir, f"training_log_{formatted_date}.jsonl")
    config_file = os.path.join(args.output_dir, f"hyperparams_{formatted_date}.json")

    # Save hyperparameters to JSON file
    hyperparams = vars(args)
    with open(config_file, "w") as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Saved configuration to: {config_file}")
    
    # Helper tensors
    one = torch.FloatTensor([1]).to(device)
    mone = one * -1
    
    # Add function to log metrics
    def log_metrics(epoch, loss_d, loss_g, lr_d=None, lr_g=None, step=None):
        log_entry = {
            "epoch": epoch,
            "loss_d": loss_d if isinstance(loss_d, float) else loss_d.item(),
            "loss_g": loss_g if isinstance(loss_g, float) else loss_g.item(),
            "lr_d": lr_d,
            "lr_g": lr_g,
            "step": step if step is not None else epoch * len(dataloader),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    # Initialize plotter
    plotter = Plotter(log_file, update_interval=5.0, left_key="loss_d", right_key="loss_g", left_label="Discriminator Loss", right_label="Generator Loss")  # Update every 5 seconds
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plot_thread.start()
    print(f"Loss plotting enabled. Progress will be saved to {os.path.join(args.output_dir, 'training_progress.png')}")

    # Training loop
    gen_iterations = 0
    global_step = 0
    progress_bar = tqdm(total=args.num_epochs * len(dataloader))
    progress_bar.set_description("Steps")
    
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Get level scenes (ignore captions if returned)
            if isinstance(batch, list):
                scenes, _ = batch  # Ignore captions
            else:
                scenes = batch
            
            scenes = scenes.to(device)
            batch_size = scenes.size(0)
            
            ############################
            # (1) Update D network
            ###########################
            # Train with real
            for p in netD.parameters():
                p.requires_grad = True
                
            # Determine number of critic iterations
            n_critic_iters = 100 if gen_iterations < 25 or gen_iterations % 500 == 0 else args.n_critic
            
            for _ in range(n_critic_iters):
                netD.zero_grad()
                
                # Clamp parameters to a range
                for p in netD.parameters():
                    p.data.clamp_(args.clamp_lower, args.clamp_upper)
                
                # Train with real data
                errD_real = netD(scenes)
                errD_real.backward(one)
                
                # Train with fake data
                noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
                with torch.no_grad():
                    fake = netG(noise)
                
                # Detach to avoid training G on these labels
                errD_fake = netD(fake.detach())
                errD_fake.backward(mone)
                
                errD = errD_real - errD_fake
                optimizerD.step()
            
            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False
                
            netG.zero_grad()
            
            # Generate fake data
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            
            # Generator wants discriminator to think it's real
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
            
            gen_iterations += 1
            
            # Update learning rate if scheduler is enabled
            if args.use_lr_scheduler:
                lr_schedulerD.step()
                lr_schedulerG.step()
                
                current_lr_d = lr_schedulerD.get_last_lr()[0]
                current_lr_g = lr_schedulerG.get_last_lr()[0]
            else:
                current_lr_d = args.lr_d
                current_lr_g = args.lr_g
            
            # Update progress bar
            progress_bar.update(1)
            logs = {
                "loss_d": errD.item(),
                "loss_g": errG.item(),
                "lr_d": current_lr_d,
                "lr_g": current_lr_g,
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            
            # Log metrics
            log_metrics(epoch, errD.item(), errG.item(), current_lr_d, current_lr_g, global_step)
            
            global_step += 1
            
            # Generate and save sample levels periodically
            if gen_iterations % 50 == 0:
                print(f'[{epoch}/{args.num_epochs}][{batch_idx}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'lr_d: {current_lr_d:.6f} lr_g: {current_lr_g:.6f}')
        
        # Generate and save sample levels at the end of each epoch
        if epoch % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            netG.eval()
            with torch.no_grad():
                fake_samples = netG(fixed_noise)
            
            # Convert samples to the right format for visualization
            # They're currently in shape [batch_size, num_tiles, height, width]
            samples_cpu = fake_samples.detach().cpu()
            
            # Visualize samples
            visualize_samples(samples_cpu, os.path.join(args.output_dir, f"samples_epoch_{epoch}"))
            netG.train()
        
        # Save checkpoints at specified intervals
        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            torch.save(netG.state_dict(), os.path.join(checkpoint_path, "generator.pth"))
            torch.save(netD.state_dict(), os.path.join(checkpoint_path, "discriminator.pth"))
            print(f"Saved models at epoch {epoch}")
    
    # Clean up plotting resources
    if plotter:
        plotter.stop_plotting()
        if plot_thread and plot_thread.is_alive():
            plot_thread.join(timeout=5.0)
    
    # Close progress bar
    progress_bar.close()
    
    # Final model save
    final_path = os.path.join(args.output_dir, "final_models")
    os.makedirs(final_path, exist_ok=True)
    torch.save(netG.state_dict(), os.path.join(final_path, "generator.pth"))
    torch.save(netD.state_dict(), os.path.join(final_path, "discriminator.pth"))
    print("Training completed. Final models saved.")

if __name__ == "__main__":
    main()