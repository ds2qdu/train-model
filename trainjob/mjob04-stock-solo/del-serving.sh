#!/bin/bash
# ============================================
# Stock Prediction Serving & Chatbot 삭제 스크립트
# ============================================

NAMESPACE="mlteam"

echo "============================================"
echo "Stock Prediction Serving & Chatbot 삭제"
echo "============================================"

kubectl delete -f 06-chatbot-ui.yaml 2>/dev/null
kubectl delete -f 05-chatbot.yaml 2>/dev/null
kubectl delete -f 04-serving.yaml 2>/dev/null

kubectl delete configmap chatbot-code -n $NAMESPACE 2>/dev/null
kubectl delete configmap chatbot-ui-code -n $NAMESPACE 2>/dev/null
kubectl delete configmap chatbot-config -n $NAMESPACE 2>/dev/null

echo ""
echo "삭제 완료!"
kubectl get pods -n $NAMESPACE | grep -E "stock|ollama"
