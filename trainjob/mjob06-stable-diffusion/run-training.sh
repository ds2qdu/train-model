#!/bin/bash
# ============================================
# Stable Diffusion - Full Deployment
# ============================================

NAMESPACE="mlteam"

echo "============================================"
echo "Stable Diffusion - Full Deployment"
echo "Image Generation with Chatbot Interface"
echo "============================================"

# 1. Create PVC
echo ""
echo "=== Step 1: Creating PVC ==="
kubectl apply -f 01-pvc.yaml

# 2. Upload scripts to PVC
echo ""
echo "=== Step 2: Uploading scripts to PVC ==="

# Create temporary pod for file copy
kubectl run sd-file-copy --image=busybox --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "copy",
        "image": "busybox",
        "command": ["sleep", "300"],
        "volumeMounts": [{
          "name": "storage",
          "mountPath": "/mnt/storage"
        }]
      }],
      "volumes": [{
        "name": "storage",
        "persistentVolumeClaim": {
          "claimName": "stable-diffusion-storage"
        }
      }]
    }
  }' -n $NAMESPACE 2>/dev/null || true

echo "Waiting for file copy pod..."
kubectl wait --for=condition=Ready pod/sd-file-copy -n $NAMESPACE --timeout=60s

# Copy files
kubectl cp train.py $NAMESPACE/sd-file-copy:/mnt/storage/train.py
kubectl cp server.py $NAMESPACE/sd-file-copy:/mnt/storage/server.py
kubectl cp chatbot.py $NAMESPACE/sd-file-copy:/mnt/storage/chatbot.py
kubectl cp chatbot_ui.py $NAMESPACE/sd-file-copy:/mnt/storage/chatbot_ui.py

# Create directories
kubectl exec sd-file-copy -n $NAMESPACE -- mkdir -p /mnt/storage/data/train /mnt/storage/models /mnt/storage/generated /mnt/storage/huggingface

# Delete temporary pod
kubectl delete pod sd-file-copy -n $NAMESPACE --wait=false

echo "Scripts uploaded to PVC"

# 3. Deploy Stable Diffusion Server
echo ""
echo "=== Step 3: Deploying SD Server ==="
kubectl apply -f 03-serving.yaml

# Wait for server to be ready
echo "Waiting for SD Server (this may take a while for model download)..."
kubectl wait --for=condition=Available deployment/stable-diffusion-server -n $NAMESPACE --timeout=600s || true

# 4. Deploy Chatbot
echo ""
echo "=== Step 4: Deploying Chatbot ==="
kubectl apply -f 04-chatbot.yaml

# 5. Check status
echo ""
echo "=== Deployment Status ==="
sleep 10

kubectl get pods -n $NAMESPACE | grep -E "(stable-diffusion|sd-)"
kubectl get svc -n $NAMESPACE | grep -E "(stable-diffusion|sd-)"

echo ""
echo "============================================"
echo "배포 완료!"
echo ""
echo "서비스 접속:"
echo "  # UI (NodePort)"
echo "  http://<node-ip>:30851"
echo ""
echo "  # Port Forward로 접속"
echo "  kubectl port-forward svc/sd-chatbot 8501:8501 -n $NAMESPACE"
echo "  http://localhost:8501"
echo ""
echo "모니터링 명령어:"
echo "  # Pod 상태 확인"
echo "  kubectl get pods -n $NAMESPACE -l app=stable-diffusion-server -w"
echo "  kubectl get pods -n $NAMESPACE -l app=sd-chatbot -w"
echo ""
echo "  # 로그 확인"
echo "  kubectl logs -f deployment/stable-diffusion-server -n $NAMESPACE"
echo "  kubectl logs -f deployment/sd-chatbot -c chatbot -n $NAMESPACE"
echo ""
echo "학습 실행 (선택사항):"
echo "  kubectl apply -f 02-training.yaml"
echo "============================================"
