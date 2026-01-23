import requests
import numpy as np
from torchvision import datasets, transforms

# KServe 엔드포인트
INFERENCE_URL = "http://mnist-classifier-predictor.mlteam.svc.cluster.local/v2/models/mnist/infer"


# MNIST 테스트셋 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.MNIST('/tmp/data', train=False, download=True, transform=transform)

def softmax(x):
    """Softmax 함수로 확률 계산"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def predict(image):
    """이미지로 추론 요청"""
    payload = {
        "inputs": [{
            "name": "input",
            "shape": [1, 1, 28, 28],
            "datatype": "FP32",
            "data": image.numpy().flatten().tolist()
        }]
    }
    
    response = requests.post(INFERENCE_URL, json=payload)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None, None
    
    result = response.json()
    logits = np.array(result["outputs"][0]["data"])
    probs = softmax(logits)
    predicted = np.argmax(probs)
    confidence = probs[predicted] * 100
    
    return predicted, confidence, probs

# === 단일 테스트 (레이블 1 찾기) ===
print("=" * 50)
print("단일 이미지 테스트")
print("=" * 50)

for i, (image, label) in enumerate(test_data):
    if label == 1:
        predicted, confidence, probs = predict(image)
        print(f"Index: {i}")
        print(f"실제 레이블: {label}")
        print(f"예측 결과: {predicted} (확률: {confidence:.2f}%)")
        print(f"전체 확률: {[f'{p*100:.1f}%' for p in probs]}")
        break

# === 여러 숫자 테스트 (0-9 각각 하나씩) ===
print("\n" + "=" * 50)
print("0-9 전체 테스트")
print("=" * 50)

tested = set()
correct = 0
total = 0

for i, (image, label) in enumerate(test_data):
    if label not in tested:
        tested.add(label)
        predicted, confidence, _ = predict(image)
        
        status = "✅" if predicted == label else "❌"
        print(f"{status} 실제: {label} → 예측: {predicted} ({confidence:.1f}%)")
        
        if predicted == label:
            correct += 1
        total += 1
        
        if len(tested) == 10:
            break

print("-" * 50)
print(f"정확도: {correct}/{total} ({correct/total*100:.0f}%)")

# === 배치 테스트 (처음 100개) ===
print("\n" + "=" * 50)
print("배치 테스트 (100개 이미지)")
print("=" * 50)

correct = 0
for i in range(100):
    image, label = test_data[i]
    predicted, _, _ = predict(image)
    if predicted == label:
        correct += 1

print(f"정확도: {correct}/100 ({correct}%)")
