#!/bin/bash
# ============================================
# Stable Diffusion - Cleanup
# ============================================

NAMESPACE="mlteam"

echo "============================================"
echo "Stable Diffusion - Cleanup"
echo "============================================"

# Delete deployments
# kubectl delete -f 04-chatbot.yaml 2>/dev/null
# kubectl delete -f 03-serving.yaml 2>/dev/null
kubectl delete -f 02-training.yaml 2>/dev/null

# Delete temporary pod if exists
# kubectl delete pod sd-file-copy -n $NAMESPACE 2>/dev/null

echo ""
echo "삭제 완료! (PVC는 유지됨)"
echo ""
echo "PVC까지 삭제하려면:"
echo "  kubectl delete -f 01-pvc.yaml"

kubectl get pods -n $NAMESPACE | grep -E "(stable-diffusion|sd-)"
