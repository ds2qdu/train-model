#!/bin/bash
set -e

# Install dependencies
echo "=== Installing dependencies ==="
pip install --root-user-action=ignore yfinance scikit-learn transformers chromadb onnxscript pytz tensorboard
echo "=== Dependencies installed successfully ==="

# Environment setup
export RANK=${JOB_COMPLETION_INDEX:-0}
export MASTER_PORT=29500
export LOCAL_RANK=0

# MASTER_ADDR: Find rank-0 pod IP via hostname pattern
# Pod naming: {trainjob}-trainer-0-{index}
MY_IP=$(hostname -i | awk '{print $1}')
MY_HOSTNAME=$(hostname)
SVC_DNS="${KUBE_TRAINJOB_NAME}.${KUBE_PROJECT}.svc.cluster.local"

# WORLD_SIZE: use KUBE_NODE_SIZE if set, otherwise count pods from DNS
if [ -n "$KUBE_NODE_SIZE" ]; then
    export WORLD_SIZE=$KUBE_NODE_SIZE
else
    echo "KUBE_NODE_SIZE not set, resolving node count from DNS..."
    for i in $(seq 1 60); do
        NODE_COUNT=$(getent ahostsv4 "$SVC_DNS" 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
        if [ "$NODE_COUNT" -ge 2 ]; then
            break
        fi
        echo "Waiting for peers... (found $NODE_COUNT, need >=2) ($i/60)"
        sleep 1
    done
    export WORLD_SIZE=$NODE_COUNT
    echo "Resolved WORLD_SIZE=$WORLD_SIZE from DNS"
fi

if echo "$MY_HOSTNAME" | grep -qE -- "-trainer-0-0$"; then
    export MASTER_ADDR=$MY_IP
    echo "I am master (rank-0): $MASTER_ADDR"
else
    echo "Waiting for master pod (rank-0)..."
    echo "My hostname: $MY_HOSTNAME, My IP: $MY_IP"
    for i in $(seq 1 60); do
        # DNS returns all pod IPs; reverse-resolve each to find -trainer-0-0-
        for ip in $(getent ahostsv4 "$SVC_DNS" 2>/dev/null | awk '{print $1}' | sort -u); do
            peer=$(getent hosts "$ip" 2>/dev/null | awk '{print $2}')
            if echo "$peer" | grep -qE -- "-trainer-0-0$"; then
                MASTER_ADDR="$ip"
                break 2
            fi
        done
        echo "Waiting... ($i/60)"
        sleep 1
    done
    echo "Found master (rank-0): $MASTER_ADDR"
    export MASTER_ADDR
fi

echo "=== Environment ==="
#echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "MY_IP=$MY_IP"
echo ""

# Start TensorBoard on Rank 0 only (background, port 6006)
if [ "$RANK" == "0" ]; then
    echo "=== Starting TensorBoard on port 6006 ==="
    tensorboard --logdir=/mnt/tensorboard --bind_all --port=6006 &
    TB_PID=$!
    echo "TensorBoard PID: $TB_PID"
fi

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
  --nnodes=$KUBE_NODE_SIZE \
  --nproc_per_node=1 \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /workspace/train.py \
  --epochs=$EPOCH_COUNT \
  --batch-size=32 \
  --lr=0.0001 \
  --seq-length=30 \
  --pred-length=5 \
  --d-model=256 \
  --nhead=8 \
  --num-layers=4 \
  --data-dir=/mnt/storage/data \
  --checkpoint-dir=/mnt/storage/checkpoints \
  --export-dir=/mnt/storage/models \
  --chromadb-dir=/mnt/storage/chromadb \
  --tensorboard-dir=/mnt/tensorboard
