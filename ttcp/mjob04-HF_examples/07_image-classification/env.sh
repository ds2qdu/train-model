#!/bin/bash
set -e

# ── Image Classification ──
# Script: examples/pytorch/image-classification/run_image_classification.py
# Model:  google/vit-base-patch16-224-in21k
# Data:   cifar10

# Install HuggingFace dependencies
echo "=== Installing HuggingFace dependencies ==="
pip install --root-user-action=ignore transformers datasets accelerate evaluate scikit-learn torchvision Pillow
echo "=== Dependencies installed successfully ==="

# Environment setup
export WORLD_SIZE=2
export RANK=${JOB_COMPLETION_INDEX:-0}
export MASTER_PORT=29500
export LOCAL_RANK=0

# MASTER_ADDR: Rank 0 uses its own IP, workers find rank 0's IP
MY_IP=$(hostname -i | awk '{print $1}')

if [ "$RANK" == "0" ]; then
    export MASTER_ADDR=$MY_IP
else
    echo "Waiting for master pod..."
    echo "My IP: $MY_IP"
    for i in $(seq 1 60); do
        MASTER_ADDR=$(getent ahostsv4 ${KUBE_TRAINJOB_NAME}.${KUBE_PROJECT}.svc.cluster.local 2>/dev/null | awk '{print $1}' | grep -v "^${MY_IP}$" | head -1)
        if [ -n "$MASTER_ADDR" ]; then
            echo "Found master: $MASTER_ADDR"
            break
        fi
        echo "Waiting... ($i/60)"
        sleep 1
    done
    export MASTER_ADDR
fi

echo "=== Environment ==="
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "MY_IP=$MY_IP"
echo ""

# Worker waits for master port
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

torchrun \
  --nnodes=$WORLD_SIZE \
  --nproc_per_node=1 \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /workspace/train.py \
  --model_name_or_path google/vit-base-patch16-224-in21k \
  --dataset_name cifar10 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_train_samples 500 \
  --max_eval_samples 500 \
  --remove_unused_columns False \
  --output_dir /mnt/storage/hf-output \
  --report_to none
