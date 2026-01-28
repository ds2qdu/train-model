#!/bin/bash
# ============================================
# Stable Diffusion - Chatbot Only Deployment
# SD Server + Chatbot (Training 없이)
# ============================================

NAMESPACE="mlteam"

echo "============================================"
echo "Stable Diffusion - Chatbot Deployment"
echo "(Training 없이 이미지 생성만)"
echo "============================================"

# 1. Create PVC
echo ""
echo "=== Creating PVC ==="
kubectl apply -f 01-pvc.yaml

# 2. Upload scripts
echo ""
echo "=== Uploading scripts ==="

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

kubectl wait --for=condition=Ready pod/sd-file-copy -n $NAMESPACE --timeout=60s

kubectl cp server.py $NAMESPACE/sd-file-copy:/mnt/storage/server.py
kubectl cp chatbot.py $NAMESPACE/sd-file-copy:/mnt/storage/chatbot.py
kubectl cp chatbot_ui.py $NAMESPACE/sd-file-copy:/mnt/storage/chatbot_ui.py

kubectl exec sd-file-copy -n $NAMESPACE -- mkdir -p /mnt/storage/models /mnt/storage/generated /mnt/storage/huggingface

kubectl delete pod sd-file-copy -n $NAMESPACE --wait=false

# 3. Deploy
echo ""
echo "=== Deploying Services ==="
kubectl apply -f 03-serving.yaml
kubectl apply -f 04-chatbot.yaml

echo ""
echo "=== Status ==="
sleep 5
kubectl get pods -n $NAMESPACE | grep -E "(stable-diffusion|sd-)"

echo ""
echo "============================================"
echo "배포 완료!"
echo ""
echo "UI 접속: http://<node-ip>:30851"
echo "또는: kubectl port-forward svc/sd-chatbot 8501:8501 -n $NAMESPACE"
echo "============================================"
