# Stock Prediction with News (Transformer + FinBERT)

## Architecture

```
[30일 가격 데이터] ──────────────────┐
                                     ├──→ Transformer ──→ 5일 예측
[30일 뉴스 데이터] ──→ FinBERT ─────┘
```

## File Structure

```
trainjob/mjob04-stock/
├── snp500-data.py        # E2E 검증 스크립트 (단일 노드)
├── train.py              # 분산학습 스크립트 (Transformer + News)
├── run-training.sh       # 학습 실행 스크립트
├── 00-resources.yaml     # Namespace, Headless Service, Queue
├── 01-pvc.yaml           # PVC
├── 02-training.yaml      # TrainingRuntime + TrainJob
├── 03-secret.yaml        # Token 
├── 04-serving.yaml       # Triton Server
├── 05-chatbot.yaml       # ollama + chatBot Backend
├── 06-chatbot-ui.yaml    # Streamlit UI
├── chatbot.py            # RAG Chatbot (Ollama)
└── chatbot_ui.py         # Streamlit UI
```

## Storage Structure

```
/mnt/storage/
├── data/
│   ├── raw_data.pt           # 원본 데이터 (증분 로딩용)
│   ├── train_data.pt         # 학습 데이터 (가격 + 뉴스 임베딩)
│   ├── test_data.pt          # 테스트 데이터
│   ├── scaler.pkl            # 가격 정규화 스케일러
│   └── last_sequence.pt      # 추론용 마지막 시퀀스
├── checkpoints/
│   ├── checkpoint_latest.pt  # 최신 체크포인트
│   └── checkpoint_best.pt    # 최고 성능 체크포인트
├── models/
│   └── stock_predictor/      # Triton 배포용
│       ├── config.pbtxt
│       └── 1/model.onnx
├── chromadb/                 # 뉴스 벡터 DB
│   └── stock_news/           # 뉴스 임베딩 컬렉션
└── huggingface/              # FinBERT 캐시
```

## Feature

| 항목 | 설명 |
|:---|:---|
| Model | Transformer (4 layers, d_model=256, 8 heads) |
| News Embedding | FinBERT (ProsusAI/finbert) |
| Data | S&P500 가격 + 영어 뉴스 (Finnhub API) |
| Input | 30일 가격 + 30일 뉴스 → 5일 예측 |
| 분산학습 | 2 Node GPU, NCCL backend |
| 증분학습 | --resume 플래그로 기존 모델 이어서 학습 |
| Vector DB | ChromaDB (뉴스 임베딩 저장/검색) |
| Export | ONNX (Triton) |

## Training

### 1. Git Push (Required)
```bash
git add trainjob/mjob04-stock/
git commit -m "Update stock training"
git push origin main
```

### 2. (Optional) Set Finnhub API Key
무료 API 키 발급: https://finnhub.io/

```yaml
# 02-training.yaml 에서 수정
- name: FINNHUB_API_KEY
  value: "your-api-key"
```

### 3. Run Training
```bash
# Option A: 스크립트 사용
chmod +x trainjob/mjob04-stock/run-training.sh
./trainjob/mjob04-stock/run-training.sh

# Option B: kubectl 직접 실행
kubectl delete trainjob stock-training -n mlteam --ignore-not-found
kubectl apply -f trainjob/mjob04-stock/00-resources.yaml
kubectl apply -f trainjob/mjob04-stock/01-pvc.yaml
kubectl apply -f trainjob/mjob04-stock/02-training.yaml
```

### 4. Monitor Training
```bash
# 상태 확인
kubectl get trainjob -n mlteam

# Pod 상태
kubectl get pods -n mlteam -w

# 로그 확인
kubectl logs -n mlteam -l jobset.sigs.k8s.io/jobset-name=stock-training -f
```

### 5. Verify Results
```bash
# 모델 파일 확인
kubectl exec -n mlteam <pod-name> -- ls -la /mnt/storage/models/stock_predictor/
```


## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  raw_data.pt 확인                                        │
│  ├── 있음 → 마지막 날짜 이후 데이터만 다운로드 (증분)     │
│  └── 없음 → 전체 10년치 다운로드 (초기)                   │
└─────────────────────────────────────────────────────────┘
    ↓
yfinance (신규 가격) + Finnhub (신규 뉴스)
    ↓
FinBERT 임베딩 (768-dim)
    ↓
┌───────────────────────────────────────┐
│ ChromaDB에 개별 뉴스 저장             │  ← RAG 검색용
│ (headline, embedding, metadata)        │
└───────────────────────────────────────┘
    ↓
기존 데이터 + 신규 데이터 병합
    ↓
raw_data.pt 저장 (다음 학습용)
    ↓
Transformer 학습
```

## Incremental Learning (증분 학습)

**데이터 증분** + **모델 증분** 두 가지가 모두 적용됩니다.

### 데이터 증분
```
1차 학습: 10년치 전체 다운로드 → raw_data.pt 저장
    ↓
2차 학습: raw_data.pt 로드 + 신규 데이터만 다운로드 → 병합
    ↓
N차 학습: 계속 누적 (효율적!)
```

### 모델 증분
`--resume` 플래그로 기존 체크포인트에서 이어서 학습합니다.
```
1차 학습: 모델 v1 저장
    ↓
2차 학습: 모델 v1 로드 → 누적 데이터로 학습 → 모델 v2 저장
```

### 전체 재학습 (처음부터)
```bash
# 데이터 + 체크포인트 모두 삭제 후 학습
kubectl exec -n mlteam <pod> -- rm -rf /mnt/storage/data/raw_data.pt
kubectl exec -n mlteam <pod> -- rm -rf /mnt/storage/checkpoints/*
./run-training.sh
```

### 모델만 재학습 (데이터 유지)
```bash
# 체크포인트만 삭제 (기존 데이터 활용)
kubectl exec -n mlteam <pod> -- rm -rf /mnt/storage/checkpoints/*
./run-training.sh
```

### TRAIN Scenario
```

```

## Model Details

### Transformer Architecture
```
Input:
  - price_input: (batch, 30, 1)    # 30일 가격
  - news_input:  (batch, 30, 768)  # 30일 뉴스 임베딩

Layers:
  - Price Embedding: Linear(1 → 256)
  - News Projection: Linear(768 → 256)
  - Positional Encoding: Learnable
  - Transformer Encoder: 4 layers, 8 heads
  - Cross-Attention: price attends to news
  - MLP Head: 256 → 128 → 5

Output:
  - (batch, 5)  # 5일 예측
```

### News Processing
```
1. Finnhub API로 뉴스 수집 (SPY, AAPL, MSFT, GOOGL + 시장 뉴스)
2. 날짜별 헤드라인 집계 (최대 5개/일)
3. FinBERT로 768차원 임베딩
4. 각 거래일에 해당 뉴스 임베딩 매핑
5. ChromaDB에 개별 뉴스 임베딩 저장 (RAG용)
```

## ChromaDB Vector Store

학습 중 수집된 뉴스 임베딩을 ChromaDB에 저장합니다. 나중에 챗봇 RAG에서 유사 뉴스 검색에 활용됩니다.

### Storage
```
/mnt/storage/chromadb/stock_news/
├── chroma.sqlite3            # 메타데이터 DB
└── embeddings/               # 벡터 저장소
```

### Collection Schema
| 필드 | 설명 |
|:---|:---|
| id | 뉴스 고유 ID (datetime_hash) |
| document | 헤드라인 텍스트 |
| embedding | FinBERT 768차원 벡터 |
| metadata.source | 뉴스 출처 |
| metadata.date | 날짜 (YYYY-MM-DD) |
| metadata.symbol | 관련 종목 |
| metadata.url | 원본 URL |

### Query Example (Python)
```python
import chromadb

client = chromadb.PersistentClient(path="/mnt/storage/chromadb")
collection = client.get_collection("stock_news")

# 유사 뉴스 검색
results = collection.query(
    query_embeddings=[your_embedding],  # 768-dim vector
    n_results=5,
    where={"date": "2025-01-23"}
)

print(results['documents'])
print(results['metadatas'])
```

## Triton Inference

### Deploy
```bash
kubectl apply -f 03-kserve.yaml  # TODO: 생성 필요
```

### Test
```python
import requests
import numpy as np

# 30일 가격 + 뉴스 임베딩 필요
price_input = np.random.randn(1, 30, 1).astype(np.float32)
news_input = np.random.randn(1, 30, 768).astype(np.float32)

response = requests.post(
    "http://localhost:8080/v2/models/stock_predictor/infer",
    json={
        "inputs": [
            {"name": "price_input", "shape": [1, 30, 1], "datatype": "FP32", "data": price_input.flatten().tolist()},
            {"name": "news_input", "shape": [1, 30, 768], "datatype": "FP32", "data": news_input.flatten().tolist()}
        ]
    }
)
print(response.json())
```

## Service Architecture

Streamlit UI (8501) -> FastAPI ChatBot (8080) -> Triton Server (8000, train model)
                                              -> ChromaDB (News RAG)
                                              -> Ollama (11434, chatbot)

## Model Role
Triton  : stock_predictor(ONNX) - 5일 가격예측 - train.py 학습
ChromaDB: FinBERT Embeded - 뉴스 유사도 검색 - train.py 저장
Ollama  : llama3.1 - 자연어대화 

## Port Forwarding
```
kubectl port-forward svc/stock-chatbot-ui 8501:8501 -n mlteam --address 0.0.0.0
```