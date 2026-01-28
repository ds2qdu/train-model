# ============================================
# YOLO Object Detection - Distributed Training
# High GPU Utilization for MLOps Validation
# ============================================

import os
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
import time
import subprocess
from datetime import datetime

def get_gpu_info():
    """Get GPU utilization and memory info"""
    if not torch.cuda.is_available():
        return "GPU: N/A"

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 4:
                    util, mem_used, mem_total, temp = parts[:4]
                    gpu_info.append(f"GPU{i}: {util}% | {mem_used}/{mem_total}MB | {temp}°C")
            return ' | '.join(gpu_info)
    except:
        pass

    # Fallback
    device = torch.cuda.current_device()
    mem_used = torch.cuda.memory_allocated(device) / 1024**3
    mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"GPU{device}: Mem {mem_used:.1f}/{mem_total:.1f}GB"

def print_gpu_status(rank, epoch=None, batch=None):
    """Print GPU status"""
    gpu_info = get_gpu_info()
    timestamp = datetime.now().strftime('%H:%M:%S')
    if epoch is not None and batch is not None:
        print(f"[{timestamp}][Rank {rank}] Epoch {epoch} Batch {batch} | {gpu_info}")
    else:
        print(f"[{timestamp}][Rank {rank}] {gpu_info}")

def setup_distributed():
    """Setup distributed training"""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        if world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            print(f"[Rank {rank}] Distributed training initialized: world_size={world_size}")
            return rank, world_size, local_rank

    return 0, 1, 0

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def download_coco_dataset(data_dir):
    """Download COCO128 dataset for training"""
    from ultralytics.utils.downloads import download

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    coco_yaml = data_path / 'coco128.yaml'

    if not (data_path / 'coco128').exists():
        print("Downloading COCO128 dataset...")
        # Download COCO128 (small version for testing)
        download('https://ultralytics.com/assets/coco128.zip', dir=data_path)
        print("Dataset downloaded successfully")

    # Create dataset yaml
    yaml_content = f"""
# COCO128 dataset
path: {data_path}/coco128
train: images/train2017
val: images/train2017

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""

    with open(coco_yaml, 'w') as f:
        f.write(yaml_content)

    return str(coco_yaml)

def train_yolo(args, rank, world_size):
    """Train YOLO model with high GPU utilization"""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  YOLO Distributed Training - Rank {rank}/{world_size}")
    print(f"{'='*60}\n")

    # Initial GPU status
    print_gpu_status(rank)

    # Download/prepare dataset
    if rank == 0:
        data_yaml = download_coco_dataset(args.data_dir)

    if world_size > 1:
        dist.barrier()  # Wait for rank 0 to download

    data_yaml = str(Path(args.data_dir) / 'coco128.yaml')

    # Load YOLO model
    # YOLOv8 sizes: n(ano), s(mall), m(edium), l(arge), x(large)
    # Larger models = more GPU usage
    model_size = args.model_size  # 'm' or 'l' for higher GPU usage

    # Check for resume options
    if args.resume:
        # Resume from specific checkpoint (additional training)
        print(f"[Rank {rank}] Loading checkpoint for additional training: {args.resume}")
        model = YOLO(args.resume)
    elif args.resume_training:
        # Resume interrupted training
        last_pt = Path(args.output_dir) / 'yolo_train_rank0' / 'weights' / 'last.pt'
        if last_pt.exists():
            print(f"[Rank {rank}] Resuming interrupted training from: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"[Rank {rank}] No checkpoint found, starting fresh")
            model = YOLO(f'yolov8{model_size}.pt')
    else:
        # Fresh training from pre-trained weights
        model = YOLO(f'yolov8{model_size}.pt')

    print(f"[Rank {rank}] Model loaded successfully")
    print_gpu_status(rank)

    # Training configuration for HIGH GPU utilization
    train_args = {
        'data': data_yaml,
        'epochs': args.epochs,
        'imgsz': args.img_size,
        'batch': args.batch_size,
        'device': rank if world_size > 1 else 0,
        'workers': 8,
        'project': args.output_dir,
        'name': f'yolo_train_rank{rank}',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'amp': True,  # Mixed precision for efficiency
        'verbose': True,
        'resume': args.resume_training,  # Resume interrupted training
    }

    # GPU monitoring callback
    class GPUMonitorCallback:
        def __init__(self, rank):
            self.rank = rank
            self.batch_count = 0

        def on_train_batch_end(self, trainer):
            self.batch_count += 1
            if self.batch_count % 10 == 0:  # Every 10 batches
                print_gpu_status(self.rank, trainer.epoch + 1, self.batch_count)

    # Start training
    print(f"\n[Rank {rank}] Starting training with config:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")
    print()

    # Monitor GPU during training
    start_time = time.time()

    try:
        results = model.train(**train_args)

        elapsed = time.time() - start_time
        print(f"\n[Rank {rank}] Training completed in {elapsed/60:.1f} minutes")
        print_gpu_status(rank)

        # Save results summary
        if rank == 0:
            summary = {
                'model': f'yolov8{model_size}',
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'img_size': args.img_size,
                'training_time_minutes': elapsed / 60,
                'world_size': world_size,
                'timestamp': datetime.now().isoformat()
            }

            import json
            summary_path = Path(args.output_dir) / 'training_summary.json'
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Training summary saved: {summary_path}")

        return results

    except Exception as e:
        print(f"[Rank {rank}] Training error: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='YOLO Distributed Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--model-size', type=str, default='m',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size (n/s/m/l/x). Larger = more GPU usage')
    parser.add_argument('--data-dir', type=str, default='/mnt/storage/data',
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default='/mnt/storage/runs',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (path to last.pt or best.pt)')
    parser.add_argument('--resume-training', action='store_true',
                        help='Resume interrupted training from last.pt')
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    print(f"\n{'='*60}")
    print(f"  YOLO Object Detection Training")
    print(f"  GPU-Intensive for MLOps Validation")
    print(f"{'='*60}")
    print(f"  Rank: {rank}/{world_size}")
    print(f"  Device: cuda:{local_rank}")
    print(f"  Model: YOLOv8{args.model_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Image Size: {args.img_size}")
    print(f"{'='*60}\n")

    try:
        train_yolo(args, rank, world_size)
    finally:
        cleanup_distributed()

    print(f"\n[Rank {rank}] Done!")

if __name__ == '__main__':
    main()
