# ============================================
# Stable Diffusion Distributed Fine-tuning
# LoRA / DreamBooth with Accelerate
#
# MLflow + k8s_metric_resolver 통합 버전
#   - ClushMLflowLogger 로 실험 정보 기록
#   - pod 이름 → k8s_metric_resolver → metric_id → MLflow tag
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

try:
    from clush_mlflow_logger import ClushMLflowLogger
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    print("[MLflow] clush_mlflow_logger not found — MLflow logging disabled", flush=True)

try:
    from k8s_metric_resolver import resolve_metric_id
    _K8S_RESOLVER_AVAILABLE = True
except ImportError:
    _K8S_RESOLVER_AVAILABLE = False
    print("[k8s_metric_resolver] module not found — k8s_metric_id tagging disabled", flush=True)


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

        if torch.isnan(torch.tensor(loss)):
            self.nan_count += 1

    def get_improvement(self):
        if self.initial_loss is None or self.initial_loss == 0:
            return 0
        return ((self.initial_loss - self.best_loss) / self.initial_loss) * 100

    def is_training_healthy(self):
        warnings = []
        if self.nan_count > 0:
            warnings.append(f"NaN loss detected {self.nan_count} times")
        if self.no_improvement_count > 20:
            warnings.append(f"No improvement for {self.no_improvement_count} epochs")
        if len(self.loss_history) > 10:
            recent_avg = sum(self.loss_history[-10:]) / 10
            early_avg = sum(self.loss_history[:10]) / min(10, len(self.loss_history))
            if recent_avg > early_avg * 1.5:
                warnings.append("Loss is increasing - possible divergence")
        if len(self.gradient_norms) > 0:
            avg_grad = sum(self.gradient_norms[-10:]) / min(10, len(self.gradient_norms))
            if avg_grad > 10:
                warnings.append(f"High gradient norm: {avg_grad:.2f}")
            if avg_grad < 1e-7:
                warnings.append(f"Vanishing gradients: {avg_grad:.2e}")
        return warnings

    def get_summary(self):
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
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def generate_sample_images(pipe, prompts, output_dir, epoch, device):
    from PIL import Image

    output_path = Path(output_dir) / "samples"
    output_path.mkdir(parents=True, exist_ok=True)

    pipe.to(device)
    pipe.safety_checker = None

    images = []
    for i, prompt in enumerate(prompts[:4]):
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
    if not torch.cuda.is_available():
        return "GPU: N/A"
    device = torch.cuda.current_device()
    mem_used = torch.cuda.memory_allocated(device) / 1024**3
    mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"GPU{device}: Mem {mem_used:.1f}/{mem_total:.1f}GB"


def get_gpu_utilization():
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 3:
                return f"GPU Util: {parts[0].strip()}%, VRAM: {parts[1].strip()}/{parts[2].strip()}MB"
    except Exception:
        pass
    return get_gpu_info()


def print_status(accelerator, message):
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime('%H:%M:%S')
        gpu_info = get_gpu_info()
        print(f"[{timestamp}] {message} | {gpu_info}", flush=True)


# ============================================
# Dataset helpers
# ============================================

def download_single_dataset(data_path, dataset_name, max_images, prefix=""):
    from PIL import Image
    from datasets import load_dataset

    count = 0

    if dataset_name == "pokemon":
        ds = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
        print(f"Pokemon dataset loaded: {len(ds)} images available", flush=True)
        for i, item in enumerate(ds):
            if i >= max_images:
                break
            img = item['image']
            caption = item['text']
            img_path = data_path / f"{prefix}pokemon_{i:04d}.png"
            img.save(img_path)
            caption_path = data_path / f"{prefix}pokemon_{i:04d}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            count += 1
            if (i + 1) % 10 == 0:
                print(f"Pokemon: {i + 1}/{min(max_images, len(ds))} images...", flush=True)

    elif dataset_name == "cat":
        ds = load_dataset("diffusers/cat_toy_example", split="train")
        print(f"Cat toy dataset loaded: {len(ds)} images available", flush=True)
        for i, item in enumerate(ds):
            if i >= max_images:
                break
            img = item['image']
            img_path = data_path / f"{prefix}cat_{i:04d}.png"
            img.save(img_path)
            caption = "a photo of sks cat toy"
            caption_path = data_path / f"{prefix}cat_{i:04d}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            count += 1

    elif dataset_name == "anime":
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
        from datasets import load_dataset_builder
        print("  Loading WikiArt label mappings...", flush=True)
        builder = load_dataset_builder("huggan/wikiart")
        features = builder.info.features
        artist_names = features['artist'].names if hasattr(features['artist'], 'names') else None
        style_names = features['style'].names if hasattr(features['style'], 'names') else None
        genre_names = features['genre'].names if hasattr(features['genre'], 'names') else None
        ds = load_dataset("huggan/wikiart", split="train", streaming=True)
        idx = 0
        for item in ds:
            if idx >= max_images:
                break
            try:
                img = item['image']
                artist_idx = item.get('artist', 0)
                style_idx = item.get('style', 0)
                genre_idx = item.get('genre', 0)
                artist = (artist_names[artist_idx] if artist_names and artist_idx < len(artist_names) else "unknown artist").replace('_', ' ')
                style  = (style_names[style_idx]   if style_names  and style_idx  < len(style_names)  else "painting").replace('_', ' ')
                genre  = (genre_names[genre_idx]   if genre_names  and genre_idx  < len(genre_names)  else "artwork").replace('_', ' ')
                img_path = data_path / f"{prefix}art_{idx:04d}.png"
                img.save(img_path)
                caption = f"a {style} {genre} painting by {artist}"
                caption_path = data_path / f"{prefix}art_{idx:04d}.txt"
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                idx += 1
                count += 1
                if idx % 50 == 0:
                    print(f"Art: {idx}/{max_images} images...", flush=True)
            except Exception:
                continue

    return count


def download_sample_dataset(data_dir, dataset_name="pokemon", max_images=50):
    from PIL import Image
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{dataset_name}' dataset to {data_dir}...", flush=True)
    try:
        from datasets import load_dataset
        if dataset_name == "all":
            all_datasets = ["pokemon", "cat", "anime", "art"]
            total_count = 0
            for ds_name in all_datasets:
                print(f"\n--- Downloading {ds_name} dataset ---", flush=True)
                try:
                    count = download_single_dataset(data_path, ds_name, max_images)
                    total_count += count
                    print(f"  {ds_name}: {count} images downloaded", flush=True)
                except Exception as e:
                    print(f"  {ds_name} failed: {e}", flush=True)
            print(f"Total: {total_count} images downloaded", flush=True)
            return True
        else:
            count = download_single_dataset(data_path, dataset_name, max_images)
            if count == 0:
                print(f"Unknown dataset: {dataset_name}. Using pokemon.", flush=True)
                return download_sample_dataset(data_dir, "pokemon", max_images)
            print(f"Dataset download complete! {count} images saved.", flush=True)
            return True
    except ImportError:
        import subprocess
        subprocess.run(['pip', 'install', 'datasets'], check=True)
        return download_sample_dataset(data_dir, dataset_name, max_images)
    except Exception as e:
        print(f"Dataset download failed: {e}", flush=True)
        return False


def prepare_dataset(data_dir, resolution, dataset_name="pokemon", max_images=50):
    from PIL import Image
    data_path = Path(data_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = []
    captions = []

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

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

    if not images:
        print(f"No images found. Downloading '{dataset_name}' dataset...", flush=True)
        if download_sample_dataset(data_dir, dataset_name, max_images):
            for img_file in data_path.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    images.append(str(img_file))
                    caption_file = img_file.with_suffix('.txt')
                    if caption_file.exists():
                        with open(caption_file, 'r', encoding='utf-8') as f:
                            captions.append(f.read().strip())
                    else:
                        captions.append(img_file.stem.replace('_', ' '))

        if not images:
            print("Download failed. Creating basic sample images...", flush=True)
            sample_colors = [
                ('red', (255, 0, 0)), ('green', (0, 255, 0)), ('blue', (0, 0, 255)),
                ('yellow', (255, 255, 0)), ('purple', (128, 0, 128)),
            ]
            for color_name, rgb in sample_colors:
                img = Image.new('RGB', (resolution, resolution), rgb)
                img_path = data_path / f"sample_{color_name}.png"
                img.save(img_path)
                images.append(str(img_path))
                captions.append(f"a solid {color_name} colored image")
                with open(data_path / f"sample_{color_name}.txt", 'w', encoding='utf-8') as f:
                    f.write(captions[-1])

    print(f"Dataset: {len(images)} images loaded from {data_dir}", flush=True)
    if images and captions:
        for i, cap in enumerate(captions[:3]):
            print(f"  {i+1}. {cap[:80]}{'...' if len(cap) > 80 else ''}", flush=True)

    return {"image_path": images, "caption": captions}


# ============================================
# LoRA Training
# ============================================
def train_lora(args, accelerator, mlflow_logger=None):
    import time as _time
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    training_start = _time.time()

    print_status(accelerator, "Loading Stable Diffusion model for LoRA training...")

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

    # ── MLflow: 모델 정보 기록 ─────────────────────────────
    if mlflow_logger is not None:
        mlflow_logger.log_model_info(args.model_name, unet)

    accelerator.wait_for_everyone()

    # LoRA config
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

    if accelerator.is_main_process:
        dataset_dict = prepare_dataset(args.train_data_dir, args.resolution, args.dataset, args.max_images)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset_dict = prepare_dataset(args.train_data_dir, args.resolution, args.dataset, args.max_images)
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
            return {"pixel_values": image, "input_ids": tokens.input_ids.squeeze()}

    train_dataset = SDDataset(dataset_dict["image_path"], dataset_dict["caption"], tokenizer, args.resolution)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ── MLflow: 데이터셋 정보 ───────────────────────────────
    if mlflow_logger is not None:
        mlflow_logger.log_dataset_info(train_dataset, train_dataset)   # val 별도 없음 → train 재사용

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    vae = vae.to(accelerator.device, dtype=torch.float16)
    text_encoder = text_encoder.to(accelerator.device, dtype=torch.float16)

    accelerator.wait_for_everyone()
    print(f"[Rank {accelerator.process_index}] Starting distributed LoRA training for {args.epochs} epochs...", flush=True)

    metrics = TrainingMetrics()
    epoch_pbar = tqdm(range(args.epochs), desc="Epochs", disable=not accelerator.is_main_process, position=0)

    for epoch in epoch_pbar:
        unet.train()
        epoch_loss = 0

        # ── MLflow: epoch 시작 ─────────────────────────────
        if mlflow_logger is not None:
            mlflow_logger.log_epoch_start()

        step_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not accelerator.is_main_process,
            position=1, leave=False
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
                grad_norm = compute_gradient_norm(unet)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            if accelerator.is_main_process:
                step_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'grad': f'{grad_norm:.2f}',
                    'gpu_mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB'
                })

        avg_loss = epoch_loss / len(train_dataloader)

        # ── 모든 rank 동기화 (MLflow/logging 전에 배리어) ────
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            metrics.update(epoch, avg_loss, grad_norm)
            for warning in metrics.is_training_healthy():
                print(warning, flush=True)

        # ── MLflow: epoch 종료 + loss 기록 ─────────────────
        if mlflow_logger is not None:
            mlflow_logger.log_epoch_end(
                epoch=epoch,
                train_loss=avg_loss,
                val_loss=avg_loss,          # diffusion 은 val set 미분리 → train loss 공유
                best_loss=metrics.best_loss,
                no_improv_count=metrics.no_improvement_count,
            )

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(f"[Rank {accelerator.process_index}] Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | {get_gpu_utilization()}", flush=True)

        if accelerator.is_main_process:
            improvement = metrics.get_improvement()
            epoch_pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'best': f'{metrics.best_loss:.4f}',
                'improve': f'{improvement:.1f}%'
            })
            print(f"\n[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} | "
                  f"Best: {metrics.best_loss:.4f} (ep{metrics.best_epoch+1}) | "
                  f"Improve: {improvement:.1f}% | {get_gpu_info()}", flush=True)

        # ── checkpoint 저장 (매 10 epoch) ──────────────────
        if (epoch + 1) % 10 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # ── MLflow: checkpoint 저장 시간 측정 ──────
                if mlflow_logger is not None:
                    mlflow_logger.log_checkpoint_start()

                unwrapped_unet = accelerator.unwrap_model(unet)
                checkpoint_path = Path(args.output_dir) / f"lora_epoch_{epoch+1}"
                unwrapped_unet.save_pretrained(checkpoint_path)
                print_status(accelerator, f"Checkpoint saved: {checkpoint_path}")

                if mlflow_logger is not None:
                    mlflow_logger.log_checkpoint_end(step=epoch)

                # 샘플 이미지 생성
                print_status(accelerator, "Generating sample images...")
                try:
                    from diffusers import StableDiffusionPipeline as _SDPipe
                    sample_pipe = _SDPipe.from_pretrained(
                        args.model_name, torch_dtype=torch.float16, safety_checker=None
                    )
                    sample_pipe.unet = unwrapped_unet
                    sample_prompts = dataset_dict["caption"][:2]
                    sample_images = generate_sample_images(
                        sample_pipe, sample_prompts, args.output_dir, epoch, accelerator.device
                    )
                    if sample_images:
                        print_status(accelerator, f"Sample images saved: {len(sample_images)} images")
                    del sample_pipe
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Sample generation skipped: {e}", flush=True)

    # ── 최종 모델 저장 ──────────────────────────────────────
    final_path = Path(args.output_dir) / "lora_final"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        final_path.mkdir(parents=True, exist_ok=True)
        unwrapped_unet.save_pretrained(final_path)

        summary = metrics.get_summary()
        training_time_sec = _time.time() - training_start

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
        print("="*60)

        # ── MLflow: sweep CSV + TTP ingest + run 종료 ───────────
        if mlflow_logger is not None:
            mlflow_logger.log_sweep_csv(
                csv_path=str(Path(args.output_dir) / "sweep_results.csv"),
                status="success",
                training_time_sec=training_time_sec,
                best_loss=summary['best_loss'],
            )
            mlflow_logger.report_to_ttp(
                status="success",
                training_time_sec=training_time_sec,
            )
            mlflow_logger.end()

        print_status(accelerator, f"LoRA training complete! Model saved to: {final_path}")

    return final_path


# ============================================
# DreamBooth Training
# ============================================
def train_dreambooth(args, accelerator, mlflow_logger=None):
    import time as _time
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    training_start = _time.time()

    print_status(accelerator, "Loading Stable Diffusion model for DreamBooth training...")

    instance_token = args.instance_token or "sks"
    class_token = args.class_token or "person"

    if accelerator.is_main_process:
        print(f"Instance token: {instance_token}", flush=True)
        print(f"Class token:    {class_token}", flush=True)

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

    # ── MLflow: 모델 정보 ───────────────────────────────────
    if mlflow_logger is not None:
        mlflow_logger.log_model_info(args.model_name, unet)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        dataset_dict = prepare_dataset(args.train_data_dir, args.resolution, args.dataset, args.max_images)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset_dict = prepare_dataset(args.train_data_dir, args.resolution, args.dataset, args.max_images)
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
                self.captions[idx], padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            )
            return {"pixel_values": image, "input_ids": tokens.input_ids.squeeze()}

    train_dataset = DreamBoothDataset(
        dataset_dict["image_path"], modified_captions, tokenizer, args.resolution
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # ── MLflow: 데이터셋 정보 ───────────────────────────────
    if mlflow_logger is not None:
        mlflow_logger.log_dataset_info(train_dataset, train_dataset)

    unet.requires_grad_(True)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    vae = vae.to(accelerator.device, dtype=torch.float16)
    text_encoder = text_encoder.to(accelerator.device, dtype=torch.float16)

    accelerator.wait_for_everyone()
    print(f"[Rank {accelerator.process_index}] Starting distributed DreamBooth training for {args.epochs} epochs...", flush=True)

    metrics = TrainingMetrics()
    epoch_pbar = tqdm(range(args.epochs), desc="Epochs", disable=not accelerator.is_main_process, position=0)

    for epoch in epoch_pbar:
        unet.train()
        epoch_loss = 0

        if mlflow_logger is not None:
            mlflow_logger.log_epoch_start()

        step_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not accelerator.is_main_process,
            position=1, leave=False
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

            if accelerator.is_main_process:
                step_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_dataloader)

        if accelerator.is_main_process:
            metrics.update(epoch, avg_loss)
            for warning in metrics.is_training_healthy():
                print(warning, flush=True)

        if mlflow_logger is not None:
            mlflow_logger.log_epoch_end(
                epoch=epoch,
                train_loss=avg_loss,
                val_loss=avg_loss,
                best_loss=metrics.best_loss,
                no_improv_count=metrics.no_improvement_count,
            )

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(f"[Rank {accelerator.process_index}] Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | {get_gpu_utilization()}", flush=True)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
            print(f"\n[Epoch {epoch+1}/{args.epochs}] Avg Loss: {avg_loss:.4f} | {get_gpu_info()}", flush=True)

    # ── 최종 모델 저장 ──────────────────────────────────────
    final_path = Path(args.output_dir) / "dreambooth_final"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path.mkdir(parents=True, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        pipe.unet = unwrapped_unet
        pipe.save_pretrained(final_path)

        summary = metrics.get_summary()
        training_time_sec = _time.time() - training_start

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

        if mlflow_logger is not None:
            mlflow_logger.log_sweep_csv(
                csv_path=str(Path(args.output_dir) / "sweep_results.csv"),
                status="success",
                training_time_sec=training_time_sec,
                best_loss=summary['best_loss'],
            )
            mlflow_logger.report_to_ttp(
                status="success",
                training_time_sec=training_time_sec,
            )
            mlflow_logger.end()

        print_status(accelerator, f"DreamBooth training complete! Model saved to: {final_path}")

    return final_path


# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Stable Diffusion Distributed Fine-tuning')

    # Model settings
    parser.add_argument('--model_name', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--train_method', type=str, default='lora', choices=['lora', 'dreambooth'])

    # Data settings
    parser.add_argument('--train_data_dir', type=str, default='/mnt/storage/data/train')
    parser.add_argument('--output_dir', type=str, default='/mnt/storage/models')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['pokemon', 'cat', 'anime', 'art', 'all'])
    parser.add_argument('--max_images', type=int, default=50)

    # Training settings
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    # LoRA specific
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_alpha', type=int, default=32)

    # DreamBooth specific
    parser.add_argument('--instance_token', type=str, default='sks')
    parser.add_argument('--class_token', type=str, default='person')

    # Resume
    parser.add_argument('--resume', type=str, default=None)

    # TensorBoard
    parser.add_argument('--tensorboard_dir', type=str, default='/mnt/tensorboard')

    # MLflow
    parser.add_argument('--mlflow_tracking_uri', type=str, default=None,
                        help='MLflow tracking URI (env MLFLOW_TRACKING_URI 로도 설정 가능)')
    parser.add_argument('--mlflow_experiment', type=str, default='stable-diffusion-finetune',
                        help='MLflow experiment name')

    args = parser.parse_args()

    # NCCL 타임아웃을 2시간으로 확장 (기본 30분 → epoch당 ~22분 학습 시 부족)
    if "NCCL_TIMEOUT" not in os.environ:
        os.environ["NCCL_TIMEOUT"] = "7200"
    # MLflow HTTP 요청 타임아웃 (기본값 없음 → 서버 무응답 시 rank 0 무한 블로킹 방지)
    if "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ:
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "30"

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=1)
    set_seed(args.seed)

    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("  Stable Diffusion Distributed Fine-tuning")
        print("="*60)
        print(f"  Model:       {args.model_name}")
        print(f"  Method:      {args.train_method}")
        print(f"  Epochs:      {args.epochs}")
        print(f"  Resolution:  {args.resolution}")
        print(f"  LR:          {args.learning_rate}")
        print(f"  Distributed: {accelerator.num_processes} processes")
        print("="*60 + "\n")

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── MLflow 초기화 (main process only) ──────────────────
    mlflow_logger = None
    if accelerator.is_main_process and _MLFLOW_AVAILABLE:
        tracking_uri = args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        mlflow_logger = ClushMLflowLogger(
            experiment_name=args.mlflow_experiment,
            run_name=f"{args.train_method}-ep{args.epochs}-lr{args.learning_rate}",
            tracking_uri=tracking_uri,
        )

        # pod_name → API → metric_id → MLflow tag
        k8s_metric_id = resolve_metric_id() if _K8S_RESOLVER_AVAILABLE else None
        mlflow_logger.start(k8s_metric_id=k8s_metric_id)

        # 하이퍼파라미터 기록
        mlflow_logger.log_hyperparams(epochs=args.epochs, batch_size=args.batch_size)
        mlflow_logger.log_gpu_info(num_gpu_nodes=accelerator.num_processes)
        _dataset_label = "all (pokemon+cat+anime+art)" if args.dataset == "all" else args.dataset
        mlflow_logger.log_early_params(model_name=args.model_name, dataset_name=_dataset_label)

    # ── 학습 실행 ───────────────────────────────────────────
    try:
        if args.train_method == 'lora':
            train_lora(args, accelerator, mlflow_logger)
        else:
            train_dreambooth(args, accelerator, mlflow_logger)
    except Exception:
        # 실패 시 TTP ingest(실패) + MLflow run 정상 종료
        if mlflow_logger is not None:
            import time as _t
            try:
                elapsed = _t.time() - (mlflow_logger._start_time or _t.time())
                mlflow_logger.report_to_ttp(status="failed", training_time_sec=elapsed)
            except Exception:
                pass
            try:
                mlflow_logger.end()
            except Exception:
                pass
        raise

    if accelerator.is_main_process:
        print("\nDistributed training completed!", flush=True)


if __name__ == '__main__':
    main()
