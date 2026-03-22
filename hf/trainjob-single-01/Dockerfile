# ============================================================
# HF TrainJob 검증용 Dockerfile
# Base: NVIDIA PyTorch (NGC) - CUDA, cuDNN, PyTorch 포함
# ============================================================

# 옵션 A: NGC PyTorch 이미지 (안정적, 드라이버 호환성 좋음)
FROM nvcr.io/nvidia/pytorch:24.01-py3

# 옵션 B: HF 공식 이미지 사용 시 아래로 교체
# FROM huggingface/transformers-pytorch-gpu:4.41.2

WORKDIR /app

# HF 라이브러리 설치
RUN pip install --no-cache-dir \
    transformers==4.44.0 \
    datasets==2.20.0 \
    accelerate==0.33.0 \
    scikit-learn \
    evaluate

# 학습 스크립트 복사
COPY 01_train.py /app/train.py

# HF 캐시 디렉토리 설정 (PVC 마운트 가능)
ENV HF_HOME=/mnt/hf-cache
ENV TRANSFORMERS_CACHE=/mnt/hf-cache

ENTRYPOINT ["python", "/app/train.py"]
