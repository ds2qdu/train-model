# ============================================
# Stable Diffusion Fine-tuning
# LoRA / DreamBooth Training
# ============================================

import os
import argparse
import torch
from pathlib import Path
from datetime import datetime
import json

def get_gpu_info():
    """Get GPU utilization info"""
    if not torch.cuda.is_available():
        return "GPU: N/A"

    device = torch.cuda.current_device()
    mem_used = torch.cuda.memory_allocated(device) / 1024**3
    mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"GPU{device}: Mem {mem_used:.1f}/{mem_total:.1f}GB"

def print_status(message):
    """Print status with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    gpu_info = get_gpu_info()
    print(f"[{timestamp}] {message} | {gpu_info}")

def prepare_dataset(data_dir, resolution):
    """Prepare training dataset"""
    from datasets import Dataset
    from PIL import Image
    import os

    data_path = Path(data_dir)

    # Check for images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = []
    captions = []

    for img_file in data_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            images.append(str(img_file))

            # Look for caption file (same name with .txt extension)
            caption_file = img_file.with_suffix('.txt')
            if caption_file.exists():
                with open(caption_file, 'r', encoding='utf-8') as f:
                    captions.append(f.read().strip())
            else:
                # Use filename as caption
                captions.append(img_file.stem.replace('_', ' '))

    if not images:
        # Create sample dataset for demo
        print("No images found. Creating sample dataset...")
        data_path.mkdir(parents=True, exist_ok=True)

        # Create a simple colored image as sample
        from PIL import Image
        for i, color in enumerate(['red', 'blue', 'green']):
            img = Image.new('RGB', (resolution, resolution), color)
            img_path = data_path / f"sample_{color}.png"
            img.save(img_path)
            images.append(str(img_path))
            captions.append(f"a {color} colored image")

        print(f"Created {len(images)} sample images")

    print(f"Dataset: {len(images)} images loaded")

    return {"image_path": images, "caption": captions}

def train_lora(args):
    """Train using LoRA (Low-Rank Adaptation)"""
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from diffusers.loaders import LoraLoaderMixin
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    print_status("Loading Stable Diffusion model for LoRA training...")

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=os.environ.get('HF_HOME', None)
    )

    # Get components
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    print_status("LoRA configured, preparing dataset...")

    # Prepare dataset
    dataset_dict = prepare_dataset(args.train_data_dir, args.resolution)

    # Custom Dataset class
    class SDDataset(Dataset):
        def __init__(self, image_paths, captions, tokenizer, resolution):
            self.image_paths = image_paths
            self.captions = captions
            self.tokenizer = tokenizer
            self.resolution = resolution

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            # Load and preprocess image
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
            image = torch.tensor(list(image.getdata())).reshape(3, self.resolution, self.resolution).float() / 127.5 - 1.0

            # Tokenize caption
            tokens = self.tokenizer(
                self.captions[idx],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )

            return {
                "pixel_values": image,
                "input_ids": tokens.input_ids.squeeze()
            }

    train_dataset = SDDataset(
        dataset_dict["image_path"],
        dataset_dict["caption"],
        tokenizer,
        args.resolution
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # Training loop
    print_status(f"Starting LoRA training for {args.epochs} epochs...")

    unet.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            input_ids = batch["input_ids"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Calculate loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                print_status(f"Epoch {epoch+1}/{args.epochs}, Step {step+1}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_dataloader)
        print_status(f"Epoch {epoch+1}/{args.epochs} completed, Avg Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(args.output_dir) / f"lora_epoch_{epoch+1}"
            unet.save_pretrained(checkpoint_path)
            print_status(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = Path(args.output_dir) / "lora_final"
    final_path.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(final_path)

    # Save training config
    config = {
        "model_name": args.model_name,
        "train_method": "lora",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "resolution": args.resolution,
        "timestamp": datetime.now().isoformat()
    }

    with open(final_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print_status(f"LoRA training complete! Model saved to: {final_path}")
    return final_path

def train_dreambooth(args):
    """Train using DreamBooth for personalization"""
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    print_status("Loading Stable Diffusion model for DreamBooth training...")

    # DreamBooth requires a unique token for the subject
    instance_token = args.instance_token or "sks"
    class_token = args.class_token or "person"

    print(f"Instance token: {instance_token}")
    print(f"Class token: {class_token}")

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=os.environ.get('HF_HOME', None)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = pipe.unet.to(device, dtype=torch.float16)
    vae = pipe.vae.to(device, dtype=torch.float16)
    text_encoder = pipe.text_encoder.to(device, dtype=torch.float16)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # Prepare dataset
    dataset_dict = prepare_dataset(args.train_data_dir, args.resolution)

    # Modify captions to include instance token
    modified_captions = [f"a photo of {instance_token} {class_token}" for _ in dataset_dict["caption"]]

    class DreamBoothDataset(Dataset):
        def __init__(self, image_paths, captions, tokenizer, resolution):
            self.image_paths = image_paths
            self.captions = captions
            self.tokenizer = tokenizer
            self.resolution = resolution

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
            image = torch.tensor(list(image.getdata())).reshape(3, self.resolution, self.resolution).float() / 127.5 - 1.0

            tokens = self.tokenizer(
                self.captions[idx],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )

            return {
                "pixel_values": image,
                "input_ids": tokens.input_ids.squeeze()
            }

    train_dataset = DreamBoothDataset(
        dataset_dict["image_path"],
        modified_captions,
        tokenizer,
        args.resolution
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Train only UNet (text encoder can also be trained but uses more memory)
    unet.requires_grad_(True)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    print_status(f"Starting DreamBooth training for {args.epochs} epochs...")

    unet.train()

    for epoch in range(args.epochs):
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % 10 == 0:
                print_status(f"Epoch {epoch+1}/{args.epochs}, Step {step+1}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_dataloader)
        print_status(f"Epoch {epoch+1}/{args.epochs} completed, Avg Loss: {avg_loss:.4f}")

    # Save model
    final_path = Path(args.output_dir) / "dreambooth_final"
    final_path.mkdir(parents=True, exist_ok=True)

    # Save the fine-tuned pipeline
    pipe.unet = unet
    pipe.save_pretrained(final_path)

    # Save config
    config = {
        "model_name": args.model_name,
        "train_method": "dreambooth",
        "instance_token": instance_token,
        "class_token": class_token,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "resolution": args.resolution,
        "timestamp": datetime.now().isoformat()
    }

    with open(final_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print_status(f"DreamBooth training complete! Model saved to: {final_path}")
    return final_path

def main():
    parser = argparse.ArgumentParser(description='Stable Diffusion Fine-tuning')

    # Model settings
    parser.add_argument('--model_name', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Base model name or path')
    parser.add_argument('--train_method', type=str, default='lora',
                        choices=['lora', 'dreambooth'],
                        help='Training method: lora or dreambooth')

    # Data settings
    parser.add_argument('--train_data_dir', type=str, default='/mnt/storage/data/train',
                        help='Training data directory')
    parser.add_argument('--output_dir', type=str, default='/mnt/storage/models',
                        help='Output directory for trained model')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Image resolution for training')

    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (1 recommended for limited GPU memory)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')

    # LoRA specific
    parser.add_argument('--lora_rank', type=int, default=4,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')

    # DreamBooth specific
    parser.add_argument('--instance_token', type=str, default='sks',
                        help='Unique token for the subject (DreamBooth)')
    parser.add_argument('--class_token', type=str, default='person',
                        help='Class token (DreamBooth)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Stable Diffusion Fine-tuning")
    print("="*60)
    print(f"  Model: {args.model_name}")
    print(f"  Method: {args.train_method}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Learning Rate: {args.learning_rate}")
    print("="*60 + "\n")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run training
    if args.train_method == 'lora':
        train_lora(args)
    else:
        train_dreambooth(args)

    print("\nTraining completed!")

if __name__ == '__main__':
    main()
