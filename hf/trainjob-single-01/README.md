# HF Trainer + Kubeflow TrainJob 검증 가이드

## 추천 조합 및 이유

| 항목 | 선택 | 이유 |
|------|------|------|
| 모델 | `distilbert-base-uncased` (66M) | 작고 빠름, GPU 1개면 충분 |
| 태스크 | SST-2 감성 분류 | HF Hub에서 바로 로드, 명확한 accuracy 메트릭 |
| 데이터셋 | GLUE SST-2 (67K train) | 작지만 실제 NLP 벤치마크 |
| 프레임워크 | HF Trainer API | 체크포인트, 메트릭, 콜백 전부 포함 |

빠른 검증 시 `--max-samples=2000` 으로 5분 내 완료 가능.
전체 데이터셋도 단일 GPU에서 ~15분이면 끝남.


## 진행 순서

### Step 1: 이미지 빌드 & 푸시

```bash
# 프로젝트 디렉토리에서
cd hf-trainjob-guide/

# 빌드
docker build -t hf-trainjob-validation:v1 .

# 로컬 레지스트리에 푸시
docker tag hf-trainjob-validation:v1 192.168.0.104:5000/hf-trainjob-validation:v1
docker push 192.168.0.104:5000/hf-trainjob-validation:v1
```

> **참고**: containerd `hosts.toml` 이 이미 10개 GPU 워커에 설정되어 있으므로
> 로컬 레지스트리에서 바로 pull 가능.


### Step 2: PVC 및 Runtime 생성

```bash
# ClusterTrainingRuntime (이미 있으면 skip)
kubectl apply -f 03_cluster-training-runtime.yaml

# PVC 생성
kubectl apply -f 02_trainjob.yaml
# (TrainJob + PVC 가 같이 들어있음)
```


### Step 3: TrainJob 실행 및 모니터링

```bash
# TrainJob 상태 확인
kubectl get trainjob hf-distilbert-sst2 -w

# Pod 확인
kubectl get pods -l job-name=hf-distilbert-sst2

# 실시간 로그
kubectl logs -f <pod-name>
```

**정상 로그 예시:**
```
=== HF TrainJob Validation ===
Model: distilbert-base-uncased
GPU available: 0
Starting training...
{'loss': 0.4523, 'learning_rate': 1.5e-05, 'epoch': 1.0}
{'eval_accuracy': 0.8945, 'eval_f1': 0.8941, 'epoch': 1.0}
...
=== Training Complete ===
Eval accuracy: 0.9105
```


### Step 4: 검증 체크리스트

TrainJob에서 HF가 "잘 동작하는지" 확인할 포인트:

#### 기본 동작
- [ ] Pod이 정상 스케줄링 됨 (KAI Scheduler 경유)
- [ ] GPU가 Pod에 정상 할당됨 (`nvidia-smi` 출력 확인)
- [ ] HF 모델/데이터셋 다운로드 성공 (인터넷 접근 or 캐시)
- [ ] 학습이 시작되고 loss가 감소함

#### 체크포인트 & 저장
- [ ] `/mnt/output/checkpoints/` 에 epoch별 체크포인트 생성됨
- [ ] `final/` 디렉토리에 최종 모델 저장됨
- [ ] PVC에 데이터가 Pod 종료 후에도 남아있음

#### GPU & 리소스
- [ ] `fp16=True` 로 mixed precision 동작 확인
- [ ] DCGM 메트릭에서 GPU utilization 확인 가능
  ```promql
  DCGM_FI_DEV_GPU_UTIL{exported_namespace="default", exported_pod=~"hf-distilbert.*"}
  ```

#### 에러 핸들링
- [ ] OOM 없이 완료 (batch_size 조정 필요 시 확인)
- [ ] TrainJob status가 Completed로 전환
- [ ] restartPolicy: OnFailure 로 일시적 실패 복구 확인


## 문제 해결 (Troubleshooting)

### HF 모델 다운로드 실패
클러스터에서 인터넷 접근이 안 되는 경우:
```bash
# 로컬에서 모델 미리 다운로드 → PVC에 복사
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('distilbert-base-uncased').save_pretrained('/tmp/distilbert')
AutoModel.from_pretrained('distilbert-base-uncased').save_pretrained('/tmp/distilbert')
"
# 이후 PVC에 복사하고 train.py에서 --model-name=/mnt/hf-cache/distilbert 로 지정
```

### GPU 할당 안 됨
```bash
# KAI Scheduler 큐 확인
kubectl get queues
kubectl describe queue <queue-name>

# GPU 리소스 확인
kubectl describe node <gpu-node> | grep -A5 "nvidia.com/gpu"
```

### Pod이 Pending 상태
```bash
kubectl describe pod <pod-name>
# Events 섹션에서 원인 확인
# - Insufficient nvidia.com/gpu → 큐 리소스 부족
# - Unschedulable → schedulerName 확인
```


## 다음 단계 (검증 통과 후)

1. **멀티 GPU 분산 학습**: `numNodes: 2` + `accelerate` 연동
2. **LoRA fine-tuning**: 더 큰 모델 (Llama 3.2 등) 에 PEFT 적용
3. **체크포인트 → KServe 배포**: ONNX export 후 InferenceService 생성
4. **Walltime 예측 연동**: PostgreSQL 에 학습 시간 기록 → backfill 스케줄링 활용
