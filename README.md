# train-model

## trainjob
Kubeflow 를 이용한 train job 샘플

### job01
sinlge GPU : 샘플 training job

### mjob02
multi GPU : 샘플 multinode training job

### mjob03
숫자 한자리 인식

# 1. 학습 하기
```
kubectl apply -f training.yaml
```

# 2. 모델 배포
```
kubectl apply -f deploy.yaml
```

# 3. 테스트 실행
```
kubectl apply -f testing.yaml
```

# Storage Structure
```
/mnt/storage/
├── checkpoints/
│   ├── checkpoint_latest.pt
│   └── checkpoint_best.pt
└── models/
    ├── model.pt          # TorchScript
    ├── model.onnx        # ONNX (Triton용)
    ├── metadata.json     # 모델 정보
    └── training_summary.json
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
