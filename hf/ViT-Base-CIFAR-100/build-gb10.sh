#!/bin/bash
# =============================================================
#  ViT-Base CIFAR-100 - GB10 DGX Spark (ARM) 이미지 빌드
# =============================================================

set -euo pipefail

REGISTRY="${REGISTRY:-registry.clush.io}"
IMAGE_NAME="${IMAGE_NAME:-mlops/vit-cifar100}"
TAG="${TAG:-gb10}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "============================================"
echo "  Building for GB10 (ARM): ${FULL_IMAGE}"
echo "============================================"

# GB10 DGX Spark 에서 직접 빌드하는 경우 (ARM 네이티브)
docker build -f Dockerfile.gb10 -t "${FULL_IMAGE}" .

# 또는 x86 호스트에서 크로스빌드하는 경우:
# docker buildx build --platform linux/arm64 -f Dockerfile.gb10 -t "${FULL_IMAGE}" --push .

docker push "${FULL_IMAGE}"

echo "============================================"
echo "  Done! Image: ${FULL_IMAGE}"
echo "============================================"
echo ""
echo "실행 순서:"
echo "  1) kubectl apply -f trainjob-1gpu-gb10.yaml"
echo "  2) kubectl -n clush-mlops get trainjob"
echo "  3) kubectl -n clush-mlops logs -f -l app=vit-cifar100"
echo ""
echo "스케일링:"
echo "  kubectl apply -f trainjob-scale-gb10.yaml  # 2, 3, 5 GPU"
