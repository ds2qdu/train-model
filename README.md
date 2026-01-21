# train-model

## trainjob
Kubeflow 를 이용한 train job 샘플

### job01
sinlge GPU : 샘플 training job

### mjob02
multi GPU : 샘플 multinode training job

### mjob03
숫자 한자리 인식

# 실행순서
```
# 1. 네임스페이스 생성 (없으면)
kubectl create namespace mlteam

# 2. PVC 생성
kubectl apply -f 01-pvc.yaml

# 3. 학습 스크립트 생성
kubectl apply -f 02-configmap.yaml

# 4. 학습 시작
kubectl apply -f 03-training.yaml

# 5. 학습 상태 확인
kubectl get trainjob -n mlteam -w

# 6. 로그 확인
kubectl logs -f -n mlteam -l training.kubeflow.org/trainjob-name=mnist-training

# 7. 학습 완료 후 모델 확인
kubectl exec -it -n mlteam <pod-name> -- ls -la /mnt/storage/models/mnist/

# 8. KServe 배포
kubectl apply -f 04-kserve.yaml

# 9. KServe 상태 확인
kubectl get inferenceservice -n mlteam -w

# 10. 배포 완료되면 테스트
kubectl apply -f 05-test.yaml
kubectl logs -f -n mlteam mnist-test
```


### KServe 상태
```
NAME               URL                                              READY   AGE
mnist-classifier   http://mnist-classifier.mlteam.example.com    True    2m
```

# File Structure
```
mnist-pipeline/
├── 01-pvc.yaml              # 저장소
├── 02-configmap.yaml        # 학습 스크립트
├── 03-training.yaml         # TrainingRuntime + TrainJob
├── 04-kserve.yaml           # KServe 배포
└── 05-test.yaml             # 추론 테스트
```

# 학습 완료후 Storage Structure
```
/mnt/storage/
├── checkpoints/
│   ├── checkpoint_latest.pt
│   └── checkpoint_best.pt
├── models/
│   ├── mnist/
│   │   ├── config.pbtxt      # Triton 설정
│   │   └── 1/
│   │       └── model.onnx    # ONNX 모델
│   ├── metadata.json
│   └── training_summary.json
```


#### Environment

##### NCCL 
NCCL 이 TCP/IP Socket 만 사용하도록 설정
NCCL_IB_DISABLE : 1 (IB 비활성화)
NCCL_SOCKET_IFNAME : eth0 (NCCL 통신 인터페이스 지정)
NCCL_NET : Socket (Use Ethernet)

#### Config
-------------------------------------------------------
setup value     | description
-------------------------------------------------------
numNodes: 2     | 2개 노드 사용
torchrun	    | PyTorch 분산 학습 런처
WORLD_SIZE,     |
    RANK,       |
    MASTER_ADDR | Kubeflow가 자동 주입하는 환경변수
NCCL_DEBUG=INFO | 노드간 통신 디버깅용
backend="nccl"  | GPU간 통신에 최적화된 백엔드
-------------------------------------------------------
