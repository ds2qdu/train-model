#!/bin/bash
set -e

# ── Translation ──
# Script: examples/pytorch/translation/run_translation.py
# Model:  Helsinki-NLP/opus-mt-en-ro
# Data:   wmt16 (en-ro)

# Install HuggingFace dependencies
echo "=== Installing HuggingFace dependencies ==="
pip install --root-user-action=ignore --upgrade transformers datasets accelerate evaluate sacrebleu sentencepiece scikit-learn
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
        MASTER_ADDR=$(getent ahostsv4 ${KUBE_TRAINJOB_NAME}-trainer-0-0.${KUBE_TRAINJOB_NAME}.${KUBE_PROJECT}.svc.cluster.local 2>/dev/null | awk '{print $1}' | head -1)
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
  --model_name_or_path Helsinki-NLP/opus-mt-en-ro \
  --dataset_name wmt16 \
  --dataset_config_name ro-en \
  --source_lang en \
  --target_lang ro \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 100 \
  --max_source_length 128 \
  --max_target_length 128 \
  --max_train_samples 200 \
  --max_eval_samples 200 \
  --predict_with_generate \
  --output_dir /mnt/storage/hf-output/04_translation \
  --report_to none
