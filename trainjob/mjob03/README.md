# train-model
Single Digit Recognition (숫자 인식)
Inference Service (Kserve  배포)

## 실행순서
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

## Triton Server 상태
```
//--- 사용가능 TAG 리스트
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags
```

## KServe 상태
```
//--- 배포상태
NAME               URL                                              READY   AGE
mnist-classifier   http://mnist-classifier.mlteam.example.com    True    2m

//--- Port-Forward
kubectl port-forward -n mlteam svc/mnist-classifier-predictor 8080:80

//--- API Connection Test
curl http://localhost:8080/v2/models/mnist

//--- 추론 테스트
curl -X POST http://localhost:8080/v2/models/mnist/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      //--- name : config.pbtxt 의 input.name 과 동일
      "name": "input",
      //--- batch, channel, height, width 순서
      //--- batch: 1개 이미지
      //--- channel: grayscale
      //--- height, width: 28x28
      "shape": [1, 1, 28, 28],
      //--- data type
      "datatype": "FP32",
      "data": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }]
  }'


-> result
{
    "model_name":"mnist",
    "model_version":"1",
    "outputs":[
        {
            "name":"output",
            "datatype":"FP32",
            "shape":[1,10],
            "data":[
                0.05898576229810715,
                0.1350831836462021,
                0.10625654458999634,
                -0.02928343415260315,
                -0.17159610986709596,
                -0.07477602362632752,
                -0.12588708102703095,
                -0.023060962557792665,
                -0.031929537653923038,
                -0.08491586893796921
                ]
        }
    ]
}
```

## kserve-test

```
pip install requests numpy torch torchvision
```

## File Structure
```
mnist-pipeline/
├── 00-resources.yaml        # namespace + Queue
├── 01-pvc.yaml              # 저장소
├── 02-configmap.yaml        # 학습 스크립트
├── 03-training.yaml         # TrainingRuntime + TrainJob
├── 04-kserve.yaml           # KServe 배포
└── 05-kserve-test.ipynb     # 추론 테스트
```

## 학습 완료후 Storage Structure
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

