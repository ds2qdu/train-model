# ============================================
# Stable Diffusion Distributed Fine-tuning
# LoRA / DreamBooth with Accelerate
# ============================================

import os
import argparse
import torch
from pathlib import Path
from datetime import datetime
import json

# Accelerate for distributed training
from accelerate import Accelerator
from accelerate.utils import set_seed

# Progress bar
from tqdm import tqdm

# ============================================
# Training Metrics Tracker
# ============================================
class TrainingMetrics:
    """Track and analyze training metrics"""
    def __init__(self):
        self.loss_history = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.initial_loss = None
        self.gradient_norms = []
        self.nan_count = 0
        self.no_improvement_count = 0

    def update(self, epoch, loss, grad_norm=None):
        """Update metrics with new epoch data"""
        self.loss_history.append(loss)

        if self.initial_loss is None:
            self.initial_loss = loss

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)

        # Check for NaN
        if torch.isnan(torch.tensor(loss)):
            self.nan_count += 1

    def get_improvement(self):
        """Get improvement percentage from initial loss"""
        if self.initial_loss is None or self.initial_loss == 0:
            return 0
        return ((self.initial_loss - self.best_loss) / self.initial_loss) * 100

    def is_training_healthy(self):
        """Check if training is progressing normally"""
        warnings = []

        if self.nan_count > 0:
            warnings.append(f"⚠️ NaN loss detected {self.nan_count} times")

        if self.no_improvement_count > 20:
            warnings.append(f"⚠️ No improvement for {self.no_improvement_count} epochs")

        if len(self.loss_history) > 10:
            recent_avg = sum(self.loss_history[-10:]) / 10
            early_avg = sum(self.loss_history[:10]) / min(10, len(self.loss_history))
            if recent_avg > early_avg * 1.5:
                warnings.append("⚠️ Loss is increasing - possible divergence")

        if len(self.gradient_norms) > 0:
            avg_grad = sum(self.gradient_norms[-10:]) / min(10, len(self.gradient_norms))
            if avg_grad > 10:
                warnings.append(f"⚠️ High gradient norm: {avg_grad:.2f}")
            if avg_grad < 1e-7:
                warnings.append(f"⚠️ Vanishing gradients: {avg_grad:.2e}")

        return warnings

    def get_summary(self):
        """Get training summary"""
        summary = {
            "initial_loss": self.initial_loss,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch + 1,
            "improvement_percent": self.get_improvement(),
            "total_epochs": len(self.loss_history),
            "nan_count": self.nan_count,
            "avg_gradient_norm": sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else None
        }
        return summary


def compute_gradient_norm(model):
    """Compute the total gradient norm of model parameters"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def generate_sample_images(pipe, prompts, output_dir, epoch, device):
    """Generate sample images to verify training quality"""
    from PIL import Image

    output_path = Path(output_dir) / "samples"
    output_path.mkdir(parents=True, exist_ok=True)

    pipe.to(device)
    pipe.safety_checker = None

    images = []
    for i, prompt in enumerate(prompts[:4]):  # Max 4 samples
        try:
            with torch.no_grad():
                image = pipe(
                    prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]

            image_path = output_path / f"epoch_{epoch+1}_sample_{i+1}.png"
            image.save(image_path)
            images.append(image_path)
        except Exception as e:
            print(f"Sample generation failed: {e}", flush=True)

    return images


def get_gpu_info():
    """Get GPU utilization info"""
    if not torch.cuda.is_available():
        return "GPU: N/A"

    device = torch.cuda.current_device()
    mem_used = torch.cuda.memory_allocated(device) / 1024**3
    mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"GPU{device}: Mem {mem_used:.1f}/{mem_total:.1f}GB"


def get_gpu_utilization():
    """Get detailed GPU utilization using nvidia-smi"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 3:
                gpu_util = parts[0].strip()
                mem_used = parts[1].strip()
                mem_total = parts[2].strip()
                return f"GPU Util: {gpu_util}%, VRAM: {mem_used}/{mem_total}MB"
    except Exception:
        pass
    return get_gpu_info()

def print_status(accelerator, message):
    """Print status with timestamp (only on main process)"""
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime('%H:%M:%S')
        gpu_info = get_gpu_info()
        print(f"[{timestamp}] {message} | {gpu_info}")

def download_single_dataset(data_path, dataset_name, max_images, prefix=""):
    """Download a single dataset from Hugging Face"""
    from PIL import Image
    from datasets import load_dataset

    count = 0

    if dataset_name == "pokemon":
        # Pokemon dataset - good for LoRA training
        ds = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
        print(f"Pokemon dataset loaded: {len(ds)} images available", flush=True)

        for i, item in enumerate(ds):
            if i >= max_images:
                break

            img = item['image']
            caption = item['text']

            # Save image
            img_path = data_path / f"{prefix}pokemon_{i:04d}.png"
            img.save(img_path)

            # Save caption
            caption_path = data_path / f"{prefix}pokemon_{i:04d}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            count += 1

            if (i + 1) % 10 == 0:
                print(f"Pokemon: {i + 1}/{min(max_images, len(ds))} images...", flush=True)

    elif dataset_name == "cat":
        # Cat toy dataset - good for DreamBooth
        ds = load_dataset("diffusers/cat_toy_example", split="train")
        print(f"Cat toy dataset loaded: {len(ds)} images available", flush=True)

        for i, item in enumerate(ds):
            if i >= max_images:
                break

            img = item['image']

            # Save image
            img_path = data_path / f"{prefix}cat_{i:04d}.png"
            img.save(img_path)

            # Create caption
            caption = "a photo of sks cat toy"
            caption_path = data_path / f"{prefix}cat_{i:04d}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            count += 1

    elif dataset_name == "anime":
        # Anime faces from a smaller dataset
        ds = load_dataset("Norod78/cartoon-blip-captions", split="train")
        print(f"Cartoon/Anime dataset loaded: {len(ds)} images available", flush=True)

        for i, item in enumerate(ds):
            if i >= max_images:
                break

            img = item['image']
            caption = item.get('text', 'an anime style illustration')

            img_path = data_path / f"{prefix}anime_{i:04d}.png"
            img.save(img_path)

            caption_path = data_path / f"{prefix}anime_{i:04d}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            count += 1

            if (i + 1) % 10 == 0:
                print(f"Anime: {i + 1}/{min(max_images, len(ds))} images...", flush=True)

    elif dataset_name == "art":
        # Art/painting dataset
        ds = load_dataset("huggan/wikiart", split="train", streaming=True)
        print(f"WikiArt dataset (streaming)...", flush=True)

        idx = 0
        for item in ds:
            if idx >= max_images:
                break
            try:
                img = item['image']
                artist = item.get('artist', 'unknown')
                style = item.get('style', 'painting')

                img_path = data_path / f"{prefix}art_{idx:04d}.png"
                img.save(img_path)

                caption = f"a {style} painting by {artist}"
                caption_path = data_path / f"{prefix}art_{idx:04d}.txt"
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

                idx += 1
                count += 1
                if idx % 10 == 0:
                    print(f"Art: {idx}/{max_images} images...", flush=True)
            except:
                continue

    return count


def download_sample_dataset(data_dir, dataset_name="pokemon", max_images=50):
    """
    Download sample dataset from Hugging Face for training.

    Available datasets:
    - pokemon: Pokemon images with captions (lambdalabs/pokemon-blip-captions)
    - cat: Cat toy images for DreamBooth (diffusers/cat_toy_example)
    - anime: Anime faces
    - art: WikiArt paintings
    - all: Download all datasets
    """
    from PIL import Image

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{dataset_name}' dataset to {data_dir}...", flush=True)

    try:
        from datasets import load_dataset

        # Download all datasets
        if dataset_name == "all":
            all_datasets = ["pokemon", "cat", "anime", "art"]
            total_count = 0

            print(f"\n{'='*50}", flush=True)
            print(f"  Downloading ALL datasets ({len(all_datasets)} types)", flush=True)
            print(f"  Max {max_images} images per dataset", flush=True)
            print(f"{'='*50}\n", flush=True)

            for ds_name in all_datasets:
                print(f"\n--- Downloading {ds_name} dataset ---", flush=True)
                try:
                    count = download_single_dataset(data_path, ds_name, max_images)
                    total_count += count
                    print(f"✓ {ds_name}: {count} images downloaded", flush=True)
                except Exception as e:
                    print(f"✗ {ds_name} failed: {e}", flush=True)

            print(f"\n{'='*50}", flush=True)
            print(f"  Total: {total_count} images downloaded", flush=True)
            print(f"{'='*50}\n", flush=True)
            return True

        else:
            # Download single dataset
            count = download_single_dataset(data_path, dataset_name, max_images)
            if count == 0:
                print(f"Unknown dataset: {dataset_name}. Using pokemon.", flush=True)
                return download_sample_dataset(data_dir, "pokemon", max_images)
            print(f"Dataset download complete! {count} images saved.", flush=True)
            return True

    except ImportError:
        print("'datasets' library not installed. Installing...", flush=True)
        import subprocess
        subprocess.run(['pip', 'install', 'datasets'], check=True)
        return download_sample_dataset(data_dir, dataset_name, max_images)

    except Exception as e:
        print(f"Dataset download failed: {e}", flush=True)
        return False


def prepare_dataset(data_dir, resolution, dataset_name="pokemon", max_images=50):
    """Prepare training dataset - downloads sample data if none exists"""
    from PIL import Image

    data_path = Path(data_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = []
    captions = []

    # 디렉토리가 없으면 생성
    if not data_path.exists():
        print(f"Data directory not found: {data_dir}", flush=True)
        data_path.mkdir(parents=True, exist_ok=True)

    # 기존 이미지 로드
    if data_path.exists():
        for img_file in data_path.iterdir():
            if img_file.suffix.lower() in image_extensions:
                images.append(str(img_file))
                caption_file = img_file.with_suffix('.txt')
                if caption_file.exists():
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        captions.append(f.read().strip())
                else:
                    captions.append(img_file.stem.replace('_', ' '))

    # 이미지가 없으면 샘플 데이터셋 다운로드
    if not images:
        print(f"No images found. Downloading '{dataset_name}' dataset...", flush=True)

        if download_sample_dataset(data_dir, dataset_name, max_images):
            # 다운로드 후 다시 로드
            for img_file in data_path.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    images.append(str(img_file))
                    caption_file = img_file.with_suffix('.txt')
                    if caption_file.exists():
                        with open(caption_file, 'r', encoding='utf-8') as f:
                            captions.append(f.read().strip())
                    else:
                        captions.append(img_file.stem.replace('_', ' '))

        # 그래도 없으면 기본 컬러 이미지 생성
        if not images:
            print("Download failed. Creating basic sample images...", flush=True)
            sample_colors = [
                ('red', (255, 0, 0)),
                ('green', (0, 255, 0)),
                ('blue', (0, 0, 255)),
                ('yellow', (255, 255, 0)),
                ('purple', (128, 0, 128)),
            ]
            for color_name, rgb in sample_colors:
                img = Image.new('RGB', (resolution, resolution), rgb)
                img_path = data_path / f"sample_{color_name}.png"
                img.save(img_path)
                images.append(str(img_path))
                caption = f"a solid {color_name} colored image"
                captions.append(caption)
                caption_file = data_path / f"sample_{color_name}.txt"
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(caption)

    print(f"Dataset: {len(images)} images loaded from {data_dir}", flush=True)

    # 샘플 캡션 출력
    if images and captions:
        print(f"Sample captions:", flush=True)
        for i, cap in enumerate(captions[:3]):
            print(f"  {i+1}. {cap[:80]}{'...' if len(cap) > 80 else ''}", flush=True)

    return {"image_path": images, "caption": captions}

def train_lora(args, accelerator):
    """Train using LoRA with distributed training"""
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    print_status(accelerator, "Loading Stable Diffusion model for LoRA training...")

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=os.environ.get('HF_HOME', None)
    )

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # Sync after model loading
    print(f"[Rank {accelerator.process_index}] Model loaded, syncing...", flush=True)
    accelerator.wait_for_everyone()

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )

    unet = get_peft_model(unet, lora_config)

    if accelerator.is_main_process:
        unet.print_trainable_parameters()

    print_status(accelerator, "LoRA configured, preparing dataset...")

    # Prepare dataset (only rank 0 downloads)
    print(f"[Rank {accelerator.process_index}] Preparing dataset...", flush=True)
    if accelerator.is_main_process:
        dataset_dict = prepare_dataset(args.train_data_dir, args.resolution, args.dataset, args.max_images)
    accelerator.wait_for_everyone()
    # All ranks load the prepared dataset
    if not accelerator.is_main_process:
        dataset_dict = prepare_dataset(args.train_data_dir, args.resolution, args.dataset, args.max_images)
    print(f"[Rank {accelerator.process_index}] Dataset ready, syncing...", flush=True)
    accelerator.wait_for_everyone()

    class SDDataset(Dataset):
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

    # Prepare for distributed training with accelerate
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Move VAE and text_encoder to device
    vae = vae.to(accelerator.device, dtype=torch.float16)
    text_encoder = text_encoder.to(accelerator.device, dtype=torch.float16)

    # Synchronize all processes before training
    print(f"[Rank {accelerator.process_index}] Waiting for all processes...", flush=True)
    accelerator.wait_for_everyone()
    print(f"[Rank {accelerator.process_index}] All processes ready!", flush=True)

    # Training loop - print from all ranks
    print(f"[Rank {accelerator.process_index}] Starting distributed LoRA training for {args.epochs} epochs...", flush=True)
    print(f"[Rank {accelerator.process_index}] World size: {accelerator.num_processes} | {get_gpu_utilization()}", flush=True)

    global_step = 0
    total_steps = len(train_dataloader) * args.epochs

    # Initialize metrics tracker
    metrics = TrainingMetrics()

    # Epoch progress bar (main process only)
    epoch_pbar = tqdm(
        range(args.epochs),
        desc="Epochs",
        disable=not accelerator.is_main_process,
        position=0
    )

    for epoch in epoch_pbar:
        unet.train()
        epoch_loss = 0

        # Step progress bar for each epoch
        step_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not accelerator.is_main_process,
            position=1,
            leave=False
        )

        for step, batch in enumerate(step_pbar):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=torch.float16)
                input_ids = batch["input_ids"]

                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                )

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
                accelerator.backward(loss)

                # Compute gradient norm before optimizer step
                grad_norm = compute_gradient_norm(unet)

                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

            # Update progress bar with loss info
            if accelerator.is_main_process:
                step_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'grad': f'{grad_norm:.2f}',
                    'gpu_mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB'
                })

        # Average loss for this rank
        avg_loss = epoch_loss / len(train_dataloader)

        # Update metrics (main process)
        if accelerator.is_main_process:
            metrics.update(epoch, avg_loss, grad_norm)

            # Check training health
            warnings = metrics.is_training_healthy()
            for warning in warnings:
                print(warning, flush=True)

        # Log from ALL ranks every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(f"[Rank {accelerator.process_index}] Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | {get_gpu_utilization()}", flush=True)

        # Sync all processes at end of epoch
        accelerator.wait_for_everyone()

        # Update epoch progress bar (main process only)
        if accelerator.is_main_process:
            improvement = metrics.get_improvement()
            epoch_pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'best': f'{metrics.best_loss:.4f}',
                'improve': f'{improvement:.1f}%'
            })
            print(f"\n[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} | "
                  f"Best: {metrics.best_loss:.4f} (ep{metrics.best_epoch+1}) | "
                  f"Improve: {improvement:.1f}% | {get_gpu_info()}")

        # Save checkpoint and generate samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            accelerator.wait_for_everyone()  # ALL processes must sync here
            if accelerator.is_main_process:
                unwrapped_unet = accelerator.unwrap_model(unet)
                checkpoint_path = Path(args.output_dir) / f"lora_epoch_{epoch+1}"
                unwrapped_unet.save_pretrained(checkpoint_path)
                print_status(accelerator, f"Checkpoint saved: {checkpoint_path}")

                # Generate sample images to verify training
                print_status(accelerator, "Generating sample images...")
                try:
                    # Create a temporary pipeline for inference
                    from diffusers import StableDiffusionPipeline
                    sample_pipe = StableDiffusionPipeline.from_pretrained(
                        args.model_name,
                        torch_dtype=torch.float16,
                        safety_checker=None
                    )
                    sample_pipe.unet = unwrapped_unet
                    sample_prompts = dataset_dict["caption"][:2]  # Use training captions
                    sample_images = generate_sample_images(
                        sample_pipe, sample_prompts, args.output_dir, epoch, accelerator.device
                    )
                    if sample_images:
                        print_status(accelerator, f"Sample images saved: {len(sample_images)} images")
                    del sample_pipe
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Sample generation skipped: {e}", flush=True)

    # Save final model
    final_path = Path(args.output_dir) / "lora_final"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        final_path.mkdir(parents=True, exist_ok=True)
        unwrapped_unet.save_pretrained(final_path)

        # Get training summary
        summary = metrics.get_summary()

        # Save training config with metrics
        config = {
            "model_name": args.model_name,
            "train_method": "lora",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "resolution": args.resolution,
            "distributed": True,
            "num_processes": accelerator.num_processes,
            "timestamp": datetime.now().isoformat(),
            "training_metrics": summary
        }

        with open(final_path / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Print comprehensive training summary
        print("\n" + "="*60)
        print("  TRAINING SUMMARY")
        print("="*60)
        print(f"  Initial Loss:     {summary['initial_loss']:.4f}")
        print(f"  Final Loss:       {summary['final_loss']:.4f}")
        print(f"  Best Loss:        {summary['best_loss']:.4f} (Epoch {summary['best_epoch']})")
        print(f"  Improvement:      {summary['improvement_percent']:.1f}%")
        print(f"  Total Epochs:     {summary['total_epochs']}")
        if summary['avg_gradient_norm']:
            print(f"  Avg Grad Norm:    {summary['avg_gradient_norm']:.4f}")
        if summary['nan_count'] > 0:
            print(f"  ⚠️ NaN Count:     {summary['nan_count']}")
        print("="*60)

        # Training quality assessment
        if summary['improvement_percent'] > 50:
            print("  ✅ Training Quality: EXCELLENT - Significant loss reduction")
        elif summary['improvement_percent'] > 20:
            print("  ✅ Training Quality: GOOD - Moderate improvement")
        elif summary['improvement_percent'] > 5:
            print("  ⚠️ Training Quality: FAIR - Minor improvement")
        else:
            print("  ❌ Training Quality: POOR - Consider adjusting hyperparameters")
        print("="*60 + "\n")

        print_status(accelerator, f"LoRA training complete! Model saved to: {final_path}")

    return final_path

def train_dreambooth(args, accelerator):
    """Train using DreamBooth with distributed training"""
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    print_status(accelerator, "Loading Stable Diffusion model for DreamBooth training...")

    instance_token = args.instance_token or "sks"
    class_token = args.class_token or "person"

    if accelerator.is_main_process:
        print(f"Instance token: {instance_token}")
        print(f"Class token: {class_token}")

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=os.environ.get('HF_HOME', None)
    )

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # Sync after model loading
    print(f"[Rank {accelerator.process_index}] Model loaded, syncing...", flush=True)
    accelerator.wait_for_everyone()

    # Prepare dataset (only rank 0 downloads)
    print(f"[Rank {accelerator.process_index}] Preparing dataset...", flush=True)
    if accelerator.is_main_process:
        dataset_dict = prepare_dataset(args.train_data_dir, args.resolution, args.dataset, args.max_images)
    accelerator.wait_for_everyone()
    # All ranks load the prepared dataset
    if not accelerator.is_main_process:
        dataset_dict = prepare_dataset(args.train_data_dir, args.resolution, args.dataset, args.max_images)
    print(f"[Rank {accelerator.process_index}] Dataset ready, syncing...", flush=True)
    accelerator.wait_for_everyone()

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

    # Optimizer
    unet.requires_grad_(True)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # Prepare for distributed training
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    vae = vae.to(accelerator.device, dtype=torch.float16)
    text_encoder = text_encoder.to(accelerator.device, dtype=torch.float16)

    # Synchronize all processes before training
    print(f"[Rank {accelerator.process_index}] Waiting for all processes...", flush=True)
    accelerator.wait_for_everyone()
    print(f"[Rank {accelerator.process_index}] All processes ready!", flush=True)

    # Training loop - print from all ranks
    print(f"[Rank {accelerator.process_index}] Starting distributed DreamBooth training for {args.epochs} epochs...", flush=True)
    print(f"[Rank {accelerator.process_index}] World size: {accelerator.num_processes} | {get_gpu_utilization()}", flush=True)

    global_step = 0
    total_steps = len(train_dataloader) * args.epochs

    # Epoch progress bar
    epoch_pbar = tqdm(
        range(args.epochs),
        desc="Epochs",
        disable=not accelerator.is_main_process,
        position=0
    )

    for epoch in epoch_pbar:
        unet.train()
        epoch_loss = 0

        # Step progress bar
        step_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not accelerator.is_main_process,
            position=1,
            leave=False
        )

        for step, batch in enumerate(step_pbar):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=torch.float16)
                input_ids = batch["input_ids"]

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

            # Update progress bar
            if accelerator.is_main_process:
                step_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'gpu_mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB'
                })

        # Average loss for this rank
        avg_loss = epoch_loss / len(train_dataloader)

        # Log from ALL ranks every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(f"[Rank {accelerator.process_index}] Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | {get_gpu_utilization()}", flush=True)

        # Sync all processes at end of epoch
        accelerator.wait_for_everyone()

        # Update epoch progress bar (main process only)
        if accelerator.is_main_process:
            epoch_pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'step': f'{global_step}/{total_steps}'
            })
            print(f"\n[Epoch {epoch+1}/{args.epochs}] Avg Loss: {avg_loss:.4f} | {get_gpu_info()}")

    # Save model
    final_path = Path(args.output_dir) / "dreambooth_final"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path.mkdir(parents=True, exist_ok=True)

        unwrapped_unet = accelerator.unwrap_model(unet)
        pipe.unet = unwrapped_unet
        pipe.save_pretrained(final_path)

        config = {
            "model_name": args.model_name,
            "train_method": "dreambooth",
            "instance_token": instance_token,
            "class_token": class_token,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "resolution": args.resolution,
            "distributed": True,
            "num_processes": accelerator.num_processes,
            "timestamp": datetime.now().isoformat()
        }

        with open(final_path / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print_status(accelerator, f"DreamBooth training complete! Model saved to: {final_path}")

    return final_path

def main():
    parser = argparse.ArgumentParser(description='Stable Diffusion Distributed Fine-tuning')

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
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['pokemon', 'cat', 'anime', 'art', 'all'],
                        help='Sample dataset to download if no images exist (all = download all 4 datasets)')
    parser.add_argument('--max_images', type=int, default=50,
                        help='Maximum number of images to download')

    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

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

    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
    )

    # Set seed for reproducibility
    set_seed(args.seed)

    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("  Stable Diffusion Distributed Fine-tuning")
        print("="*60)
        print(f"  Model: {args.model_name}")
        print(f"  Method: {args.train_method}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Resolution: {args.resolution}")
        print(f"  Learning Rate: {args.learning_rate}")
        print(f"  Distributed: {accelerator.num_processes} processes")
        print(f"  Process Index: {accelerator.process_index}")
        print("="*60 + "\n")

    # Create output directory
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run training
    if args.train_method == 'lora':
        train_lora(args, accelerator)
    else:
        train_dreambooth(args, accelerator)

    if accelerator.is_main_process:
        print("\nDistributed training completed!")

if __name__ == '__main__':
    main()
