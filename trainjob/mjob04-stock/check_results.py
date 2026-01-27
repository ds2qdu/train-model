# ============================================
# 학습 결과 확인 스크립트
# ============================================

import torch
import json
import pickle
from pathlib import Path
from datetime import datetime

# PVC 마운트 경로 설정
PVC_PATH = Path("y:/mlteam-stock-pipeline-storage-pvc-fab68a2b-8953-4b90-a164-6fdf446c9836")

def check_training_summary():
    """학습 결과 요약 확인"""
    print("=" * 60)
    print("1. 학습 결과 요약 (training_summary.json)")
    print("=" * 60)

    summary_path = PVC_PATH / "models" / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        print("  파일 없음")

def check_checkpoint():
    """체크포인트 정보 확인"""
    print("\n" + "=" * 60)
    print("2. 체크포인트 정보")
    print("=" * 60)

    for ckpt_name in ['checkpoint_best.pt', 'checkpoint_latest.pt']:
        ckpt_path = PVC_PATH / "checkpoints" / ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            print(f"\n  [{ckpt_name}]")
            print(f"    Epoch: {ckpt.get('epoch', 'N/A') + 1}")
            print(f"    Loss: {ckpt.get('loss', 'N/A'):.6f}")
            print(f"    Best Loss: {ckpt.get('best_loss', 'N/A'):.6f}")
            print(f"    Timestamp: {ckpt.get('timestamp', 'N/A')}")

def check_raw_data():
    """원본 데이터 정보 확인"""
    print("\n" + "=" * 60)
    print("3. 학습 데이터 정보 (raw_data.pt)")
    print("=" * 60)

    raw_data_path = PVC_PATH / "data" / "raw_data.pt"
    if raw_data_path.exists():
        data = torch.load(raw_data_path, map_location='cpu', weights_only=False)
        print(f"  총 일수: {data.get('num_days', 'N/A')}")
        print(f"  마지막 날짜: {data.get('last_date', 'N/A')}")
        print(f"  업데이트 시간: {data.get('updated_at', 'N/A')}")
        print(f"  가격 데이터 shape: {data.get('prices', []).shape if hasattr(data.get('prices', []), 'shape') else len(data.get('prices', []))}")
        print(f"  뉴스 임베딩 shape: {data.get('news_embeddings', torch.tensor([])).shape}")

def check_chromadb():
    """ChromaDB 벡터 데이터베이스 정보 확인"""
    print("\n" + "=" * 60)
    print("4. ChromaDB 벡터 데이터베이스")
    print("=" * 60)

    chromadb_path = PVC_PATH / "chromadb"
    if chromadb_path.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chromadb_path))
            collection = client.get_collection("stock_news")

            count = collection.count()
            print(f"  Collection: stock_news")
            print(f"  총 문서 수: {count}")

            # 샘플 데이터 조회
            if count > 0:
                sample = collection.peek(limit=3)
                print(f"\n  [최근 뉴스 샘플 (3개)]")
                for i, doc in enumerate(sample['documents'][:3]):
                    meta = sample['metadatas'][i] if sample['metadatas'] else {}
                    print(f"    {i+1}. [{meta.get('date', 'N/A')}] {doc[:80]}...")
        except ImportError:
            print("  chromadb 모듈이 설치되지 않음. pip install chromadb")
        except Exception as e:
            print(f"  에러: {e}")
    else:
        print("  ChromaDB 폴더 없음")

def check_model_files():
    """모델 파일 확인"""
    print("\n" + "=" * 60)
    print("5. 내보낸 모델 파일")
    print("=" * 60)

    models_path = PVC_PATH / "models"

    # ONNX 모델
    onnx_path = models_path / "stock_predictor" / "1" / "model.onnx"
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX 모델: {onnx_path.name} ({size_mb:.1f} MB)")
    else:
        # PyTorch fallback
        pt_path = models_path / "stock_predictor" / "1" / "model.pt"
        if pt_path.exists():
            size_mb = pt_path.stat().st_size / (1024 * 1024)
            print(f"  PyTorch 모델: {pt_path.name} ({size_mb:.1f} MB)")
        else:
            print("  모델 파일 없음")

    # Triton config
    config_path = models_path / "stock_predictor" / "config.pbtxt"
    if config_path.exists():
        print(f"  Triton config: {config_path.name}")

def compare_with_previous():
    """이전 결과와 비교 (raw_data.pt의 변경 추적)"""
    print("\n" + "=" * 60)
    print("6. 변경 이력 확인")
    print("=" * 60)

    raw_data_path = PVC_PATH / "data" / "raw_data.pt"
    if raw_data_path.exists():
        data = torch.load(raw_data_path, map_location='cpu', weights_only=False)

        # 날짜 범위 확인
        dates = data.get('dates', [])
        if dates:
            print(f"  데이터 시작일: {dates[0]}")
            print(f"  데이터 종료일: {dates[-1]}")
            print(f"  총 거래일: {len(dates)}일")

            # 최근 추가된 데이터 (마지막 10일)
            print(f"\n  [최근 10일 데이터]")
            for date in dates[-10:]:
                print(f"    - {date}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Stock Prediction Model - 학습 결과 확인")
    print("=" * 60)

    check_training_summary()
    check_checkpoint()
    check_raw_data()
    check_chromadb()
    check_model_files()
    compare_with_previous()

    print("\n" + "=" * 60)
    print("  확인 완료")
    print("=" * 60)
