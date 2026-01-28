# Create Image 

## Architecture

```
  [User] → [Streamlit UI:8501] → [Chatbot:8080] → [Ollama LLM]
                                        ↓
                                [SD Server:8000] → [Stable Diffusion Model]
                                        ↓
                                [Generated Images]
```

## File Structure

```
  mjob06-stable-diffusion/
  ├── 01-pvc.yaml           # 100Gi 스토리지 (모델, 이미지 저장)
  ├── 02-training.yaml      # Fine-tuning TrainJob (LoRA/DreamBooth)
  ├── train.py              # 학습 스크립트
  ├── 03-serving.yaml       # 이미지 생성 서버 (FastAPI)
  ├── server.py             # SD 서버 코드
  ├── 04-chatbot.yaml       # 챗봇 (Ollama + UI)
  ├── chatbot.py            # 챗봇 백엔드
  ├── chatbot_ui.py         # Streamlit UI
  ├── run-training.sh       # 전체 배포 스크립트
  ├── run-chatbot-only.sh   # 챗봇만 배포 (학습 없이)
  └── del.sh                # 정리 스크립트
```

## Fine-Tuning


## Training Data
LoRA : 스타일 학습
DreamBooth : 캐린터/인물 학습


```
  /mnt/storage/data/train/
  ├── image1.png
  ├── image1.txt    # "a photo of sks character"
  ├── image2.png
  └── image2.txt    # "a landscape painting in mystic style, mountains and lake"

  sks = unique token
```


## How To Deploy

### Deploy ChatBot (Create Image)
```
run-chatbot-only.sh
```

### Deploy All (including training)
```
run-training.sh
```

### Connect UI
```
http://<node-ip>:30851
```
