#!/bin/bash
set -e

# ============================================================
# Stable Diffusion LoRA/DreamBooth — TrainJob entrypoint
# - accelerate launch (분산 학습)
# - MLflow + k8s_metric_resolver 통합
# ============================================================

# ── git repo에서 모듈 복사 ────────────────────────────────────
echo "=== Copying modules from git repo ==="
cp /tmp/repo/ttcp/mjob02-diffusion/train.py /workspace/train.py
cp /tmp/repo/ttcp/mjob02-diffusion/clush_mlflow_logger.py /workspace/clush_mlflow_logger.py
cp /tmp/repo/ttcp/mjob02-diffusion/k8s_metric_resolver.py /workspace/k8s_metric_resolver.py
echo "=== Modules copied ==="

# ── 의존 패키지 설치 ─────────────────────────────────────────
echo "=== Installing Stable Diffusion dependencies ==="
pip install --root-user-action=ignore \
  diffusers[torch] \
  transformers \
  accelerate \
  safetensors \
  peft \
  datasets \
  Pillow \
  ftfy \
  tqdm \
  mlflow \
  psycopg2-binary
echo "=== Dependencies installed ==="

# ── huggingface 캐시 경로 ────────────────────────────────────
export HF_HOME=/mnt/storage/huggingface
export TRANSFORMERS_CACHE=/mnt/storage/huggingface

# ── 분산 학습 환경 변수 ──────────────────────────────────────
export WORLD_SIZE=2
export RANK=${JOB_COMPLETION_INDEX:-0}
export MASTER_PORT=29500
export LOCAL_RANK=0

MY_IP=$(hostname -i | awk '{print $1}')

if [ "$RANK" == "0" ]; then
    export MASTER_ADDR=$MY_IP
else
    echo "Waiting for master pod..."
    echo "My IP: $MY_IP"
    for i in $(seq 1 60); do
        MASTER_ADDR=$(getent ahostsv4 ${KUBE_TRAINJOB_NAME}.${KUBE_PROJECT}.svc.cluster.local 2>/dev/null \
                        | awk '{print $1}' | grep -v "^${MY_IP}$" | head -1)
        if [ -n "$MASTER_ADDR" ]; then
            echo "Found master: $MASTER_ADDR"
            break
        fi
        echo "Waiting... ($i/60)"
        sleep 1
    done
    export MASTER_ADDR
fi

echo "=== Distributed Training Config ==="
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "MY_IP=$MY_IP"
echo ""

echo "=== GPU Information ==="
nvidia-smi

# ── TensorBoard (Rank 0 only) ────────────────────────────────
if [ "$RANK" == "0" ]; then
    echo "=== Starting TensorBoard on port 6006 ==="
    tensorboard --logdir=/mnt/tensorboard --bind_all --port=6006 &
    TB_PID=$!
    echo "TensorBoard PID: $TB_PID"
fi

# ── Worker : master port 열릴 때까지 대기 ────────────────────
if [ "$RANK" != "0" ]; then
    echo "Waiting for master port..."
    for i in $(seq 1 120); do
        if timeout 1 bash -c "</dev/tcp/$MASTER_ADDR/$MASTER_PORT" 2>/dev/null; then
            echo "Master is ready!"
            break
        fi
        sleep 1
    done
fi

# ── accelerate launch ────────────────────────────────────────
echo "=== Starting Distributed Stable Diffusion Fine-tuning ==="
cd /mnt/storage

accelerate launch \
  --num_processes=$WORLD_SIZE \
  --num_machines=$WORLD_SIZE \
  --machine_rank=$RANK \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=fp16 \
  --dynamo_backend=no \
  /workspace/train.py \
  --model_name "runwayml/stable-diffusion-v1-5" \
  --train_data_dir /mnt/storage/data/train \
  --output_dir /mnt/storage/models \
  --tensorboard_dir /mnt/tensorboard \
  --train_method lora \
  --epochs 100 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --resolution 512 \
  --dataset all \
  --max_images 3000 \
  --mlflow_tracking_uri "${MLFLOW_TRACKING_URI:-http://192.168.0.123:30500}" \
  --mlflow_experiment "stable-diffusion-finetune"

echo "=== Training Complete ==="
