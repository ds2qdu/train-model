#!/bin/bash
# ============================================
# YOLO Training - Cleanup
# ============================================

NAMESPACE="mlteam"

echo "============================================"
echo "YOLO Training - Cleanup"
echo "============================================"

kubectl delete -f 02-training.yaml 2>/dev/null
kubectl delete pod yolo-file-copy -n $NAMESPACE 2>/dev/null

echo ""
echo "삭제 완료! (PVC는 유지됨)"
echo ""
echo "PVC까지 삭제하려면:"
echo "  kubectl delete -f 01-pvc.yaml"

kubectl get pods -n $NAMESPACE | grep yolo
