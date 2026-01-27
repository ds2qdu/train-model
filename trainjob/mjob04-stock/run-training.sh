#!/bin/bash
# Stock Prediction Model Training Script
# Usage: ./run-training.sh

set -e

NAMESPACE="mlteam"
TRAINJOB_NAME="stock-training"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "Stock Prediction Model Training"
echo "========================================"
echo ""

# 1. Check if previous TrainJob exists
if kubectl get trainjob ${TRAINJOB_NAME} -n ${NAMESPACE} &>/dev/null; then
    echo "Previous TrainJob found. Deleting..."
    kubectl delete trainjob ${TRAINJOB_NAME} -n ${NAMESPACE}
    echo "Waiting for cleanup..."
    sleep 5
fi

# 2. Apply resources (idempotent)
echo ""
echo "Applying resources..."
kubectl apply -f ${SCRIPT_DIR}/00-resources.yaml
kubectl apply -f ${SCRIPT_DIR}/01-pvc.yaml
kubectl apply -f ${SCRIPT_DIR}/03-secret.yaml
kubectl apply -f ${SCRIPT_DIR}/02-training.yaml

# 3. Wait for pods to start
echo ""
echo "Waiting for training pods..."
sleep 10

# 4. Show status
echo ""
echo "Training started!"
echo "========================================"
kubectl get pods -n ${NAMESPACE} -l jobset.sigs.k8s.io/jobset-name=${TRAINJOB_NAME}

echo ""
echo "To watch logs:"
echo "  kubectl logs -n ${NAMESPACE} -l jobset.sigs.k8s.io/jobset-name=${TRAINJOB_NAME} -f"
echo ""
echo "To check status:"
echo "  kubectl get trainjob -n ${NAMESPACE}"
echo ""
