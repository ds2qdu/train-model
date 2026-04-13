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

# [DEBUG] Dump all KUBE_* env vars to check what the container actually received
echo "[DEBUG] === KUBE env vars ==="
env | grep -E "^KUBE_" | sort || echo "[DEBUG] No KUBE_* env vars found"
echo "[DEBUG] KUBE_NODE_SIZE raw value: '${KUBE_NODE_SIZE}'"
echo "[DEBUG] KUBE_TRAINJOB_NAME raw value: '${KUBE_TRAINJOB_NAME}'"
echo "[DEBUG] KUBE_PROJECT raw value: '${KUBE_PROJECT}'"
echo "[DEBUG] JOB_COMPLETION_INDEX raw value: '${JOB_COMPLETION_INDEX}'"
echo "[DEBUG] === All env vars ==="
env | sort
echo "[DEBUG] ====================="


if echo "$MY_HOSTNAME" | grep -qE -- "-trainer-0-0$"; then
    export MASTER_ADDR=$MY_IP
    echo "I am master (rank-0): $MASTER_ADDR"
else
    echo "Waiting for master pod (rank-0)..."
    echo "My hostname: $MY_HOSTNAME, My IP: $MY_IP"
    echo "[DEBUG] SVC_DNS=$SVC_DNS"
    for i in $(seq 1 60); do
        # DNS returns all pod IPs; reverse-resolve each to find -trainer-0-0
        ALL_IPS=$(getent ahostsv4 "$SVC_DNS" 2>/dev/null | awk '{print $1}' | sort -u)
        echo "[DEBUG] DNS lookup for $SVC_DNS returned IPs: $(echo $ALL_IPS | tr '\n' ' ')"
        for ip in $ALL_IPS; do
            peer_full=$(getent hosts "$ip" 2>/dev/null)
            peer=$(echo "$peer_full" | awk '{print $2}')
            echo "[DEBUG] reverse lookup ip=$ip => raw='$peer_full' => hostname='$peer'"
            if echo "$peer" | grep -qE -- "-trainer-0-0(\.|$)"; then
                MASTER_ADDR="$ip"
                echo "[DEBUG] MATCHED master! ip=$ip hostname=$peer"
                break 2
            fi
        done
        echo "Waiting... ($i/60)"
        sleep 1
    done
    echo "Found master (rank-0): MASTER_ADDR='$MASTER_ADDR'"
    export MASTER_ADDR
fi

echo "=== Environment ==="
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
