#!/bin/bash
# ============================================
# Stock Prediction Serving & Chatbot 배포 스크립트
# Local LLM (Ollama) 버전
# ============================================

NAMESPACE="mlteam"

echo "============================================"
echo "Stock Prediction Serving & Chatbot 배포"
echo "Local LLM (Ollama) Version"
echo "============================================"

# 1. ConfigMap으로 코드 배포
echo ""
echo "=== Creating ConfigMaps for code ==="

# Chatbot code
kubectl delete configmap chatbot-code -n $NAMESPACE 2>/dev/null
kubectl create configmap chatbot-code \
    --from-file=chatbot.py \
    -n $NAMESPACE

# Chatbot UI code
kubectl delete configmap chatbot-ui-code -n $NAMESPACE 2>/dev/null
kubectl create configmap chatbot-ui-code \
    --from-file=chatbot_ui.py \
    -n $NAMESPACE

echo "ConfigMaps created."

# 2. Triton Inference Server 배포
echo ""
echo "=== Deploying Triton Inference Server ==="
kubectl apply -f 04-serving.yaml

# 3. Ollama + Chatbot Backend 배포
echo ""
echo "=== Deploying Ollama LLM & Chatbot Backend ==="
kubectl apply -f 05-chatbot.yaml

# 4. Chatbot UI 배포
echo ""
echo "=== Deploying Chatbot UI ==="
kubectl apply -f 06-chatbot-ui.yaml

# 5. 상태 확인
echo ""
echo "=== Deployment Status ==="
echo ""
echo "Waiting for pods to start..."
sleep 10

echo ""
echo "--- Triton Server ---"
kubectl get pods -n $NAMESPACE -l app=stock-predictor-triton

echo ""
echo "--- Ollama LLM ---"
kubectl get pods -n $NAMESPACE -l app=ollama

echo ""
echo "--- Chatbot Backend ---"
kubectl get pods -n $NAMESPACE -l app=stock-chatbot

echo ""
echo "--- Chatbot UI ---"
kubectl get pods -n $NAMESPACE -l app=stock-chatbot-ui

echo ""
echo "=== Services ==="
kubectl get svc -n $NAMESPACE | grep -E "stock|ollama"

echo ""
echo "============================================"
echo "배포 완료!"
echo ""
echo "※ Ollama 모델 다운로드에 시간이 걸릴 수 있습니다."
echo "   진행 상황 확인: kubectl logs -f -l app=ollama -n $NAMESPACE"
echo ""
echo "접속 방법:"
echo "  1. Triton Server:  kubectl port-forward svc/stock-predictor-triton 8000:8000 -n $NAMESPACE"
echo "  2. Ollama LLM:     kubectl port-forward svc/ollama 11434:11434 -n $NAMESPACE"
echo "  3. Chatbot API:    kubectl port-forward svc/stock-chatbot 8080:8080 -n $NAMESPACE"
echo "  4. Chatbot UI:     kubectl port-forward svc/stock-chatbot-ui 8501:8501 -n $NAMESPACE"
echo ""
echo "또는 Ingress 사용시:"
echo "  - Triton:    http://stock-predictor.local"
echo "  - API:       http://stock-chatbot.local"
echo "  - UI:        http://stock-ui.local"
echo "============================================"
