#!/bin/bash
# ============================================
# YOLO Object Detection Training - Deploy
# ============================================

NAMESPACE="mlteam"

echo "============================================"
echo "YOLO Object Detection - Distributed Training"
echo "2 Nodes GPU Training for MLOps Validation"
echo "============================================"

# 1. Create PVC
echo ""
echo "=== Creating PVC ==="
kubectl apply -f 01-pvc.yaml

# 2. Upload train.py to PVC
echo ""
echo "=== Uploading train.py to PVC ==="

# Create a temporary pod to copy files
kubectl run yolo-file-copy --image=busybox --restart=Never \
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
          "claimName": "yolo-training-storage"
        }
      }]
    }
  }' -n $NAMESPACE 2>/dev/null || true

echo "Waiting for file copy pod..."
kubectl wait --for=condition=Ready pod/yolo-file-copy -n $NAMESPACE --timeout=60s

# Copy train.py
kubectl cp train.py $NAMESPACE/yolo-file-copy:/mnt/storage/train.py

# Create directories
kubectl exec yolo-file-copy -n $NAMESPACE -- mkdir -p /mnt/storage/data /mnt/storage/runs

# Delete temporary pod
kubectl delete pod yolo-file-copy -n $NAMESPACE --wait=false

echo "train.py uploaded to PVC"

# 3. Deploy TrainJob
echo ""
echo "=== Deploying TrainJob ==="
kubectl apply -f 02-training.yaml

# 4. Check status
echo ""
echo "=== Deployment Status ==="
sleep 5

kubectl get trainjob -n $NAMESPACE | grep yolo
kubectl get pods -n $NAMESPACE | grep yolo

echo ""
echo "============================================"
echo "배포 완료!"
echo ""
echo "모니터링 명령어:"
echo "  # Pod 상태 확인"
echo "  kubectl get pods -n $NAMESPACE -l job-name=yolo-distributed-training -w"
echo ""
echo "  # 학습 로그 확인"
echo "  kubectl logs -f -l job-name=yolo-distributed-training -n $NAMESPACE"
echo ""
echo "  # GPU 사용률 확인 (Pod 내부)"
echo "  kubectl exec -it <pod-name> -n $NAMESPACE -- nvidia-smi -l 1"
echo ""
echo "============================================"
