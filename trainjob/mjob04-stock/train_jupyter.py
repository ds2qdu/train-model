# ============================================
# S&P 500 Stock Prediction - Transformer + News
# Jupyter Notebook Version (Single GPU/CPU)
# ============================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import json
import pickle
import requests
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Configuration (대신 argparse 대체)
# ============================================
class Args:
    epochs = 10              # Jupyter 테스트용으로 줄임
    batch_size = 32
    lr = 0.0001
    seq_length = 30
    pred_length = 5
    d_model = 256
    nhead = 8
    num_layers = 4
    data_dir = '/tmp/stock_data'
    checkpoint_dir = '/tmp/stock_checkpoints'
    export_dir = '/tmp/stock_models'
    chromadb_dir = '/tmp/stock_chromadb'
    resume = False

args = Args()

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# News Data Collection (Finnhub)
# ============================================
class NewsCollector:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('FINNHUB_API_KEY', 'd5rfmh1r01qunvpsi5egd5rfmh1r01qunvpsi5f0')
        self.base_url = "https://finnhub.io/api/v1"

    def get_news(self, symbol="SPY", from_date=None, to_date=None):
        """Fetch news for a symbol within date range"""
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')

        url = f"{self.base_url}/company-news"
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
            'token': self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                news_list = response.json()
                return news_list
            else:
                print(f"News API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"News fetch error: {e}")
            return []

    def get_market_news(self, category="general"):
        """Fetch general market news"""
        url = f"{self.base_url}/news"
        params = {
            'category': category,
            'token': self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"Market news fetch error: {e}")
            return []

# ============================================
# News Embedding (FinBERT)
# ============================================
class NewsEmbedder:
    def __init__(self, model_name="ProsusAI/finbert", device='cuda'):
        self.device = device
        print(f"Loading FinBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.embedding_dim = 768

    def embed(self, texts, max_length=128):
        """Convert list of texts to embeddings"""
        if not texts:
            return torch.zeros(1, self.embedding_dim)

        embeddings = []
        with torch.no_grad():
            for text in texts:
                if not text:
                    embeddings.append(torch.zeros(self.embedding_dim))
                    continue

                inputs = self.tokenizer(
                    text[:512],
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                ).to(self.device)

                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(cls_embedding.cpu())

        return torch.stack(embeddings) if embeddings else torch.zeros(1, self.embedding_dim)

# ============================================
# ChromaDB Vector Store
# ============================================
class NewsVectorStore:
    def __init__(self, persist_dir="/tmp/chromadb"):
        """Initialize ChromaDB for news embedding storage"""
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="stock_news",
            metadata={"description": "Financial news embeddings for stock prediction"}
        )
        print(f"ChromaDB initialized at {persist_dir}")
        print(f"Collection 'stock_news' contains {self.collection.count()} documents")

    def add_news(self, news_items, embeddings):
        if len(news_items) == 0:
            return

        ids = []
        documents = []
        metadatas = []
        embedding_list = []
        seen_ids = set()  # 배치 내 중복 체크용

        for i, news in enumerate(news_items):
            news_datetime = news.get('datetime', 0)
            headline = news.get('headline', '')
            # 더 고유한 ID 생성: datetime + headline hash + index
            news_id = f"{news_datetime}_{hash(headline) % 1000000}_{i}"

            # 배치 내 중복 체크
            if news_id in seen_ids:
                continue
            seen_ids.add(news_id)

            # DB 내 중복 체크
            try:
                existing = self.collection.get(ids=[news_id])
                if existing and len(existing['ids']) > 0:
                    continue
            except:
                pass

            ids.append(news_id)
            documents.append(headline)
            metadatas.append({
                'source': news.get('source', 'unknown'),
                'datetime': str(news_datetime),
                'date': datetime.fromtimestamp(news_datetime).strftime('%Y-%m-%d') if news_datetime else '',
                'url': news.get('url', ''),
                'symbol': news.get('related', news.get('symbol', '')),
                'category': news.get('category', 'company'),
                'summary': news.get('summary', '')[:500] if news.get('summary') else ''
            })
            embedding_list.append(embeddings[i].tolist() if torch.is_tensor(embeddings[i]) else embeddings[i])

        if ids:
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embedding_list
                )
                print(f"Added {len(ids)} news items to ChromaDB")
            except Exception as e:
                print(f"Warning: ChromaDB add failed: {e}")

    def get_by_date(self, date):
        """Get all news for a specific date"""
        results = self.collection.get(
            where={"date": date},
            include=["embeddings", "documents", "metadatas"]
        )
        return results

    def get_stats(self):
        return {
            'total_documents': self.collection.count(),
            'persist_dir': self.persist_dir
        }

# ============================================
# Dataset
# ============================================
class StockNewsDataset(Dataset):
    def __init__(self, price_sequences, news_embeddings, targets):
        self.price_sequences = price_sequences
        self.news_embeddings = news_embeddings
        self.targets = targets

    def __len__(self):
        return len(self.price_sequences)

    def __getitem__(self, idx):
        return (
            self.price_sequences[idx],
            self.news_embeddings[idx],
            self.targets[idx]
        )

# ============================================
# Transformer Model
# ============================================
class StockNewsTransformer(nn.Module):
    def __init__(
        self,
        price_dim=1,
        news_dim=768,
        d_model=256,
        nhead=8,
        num_layers=4,
        seq_length=30,
        pred_length=5,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length

        # Price embedding
        self.price_embedding = nn.Linear(price_dim, d_model)

        # News embedding projection
        self.news_projection = nn.Linear(news_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention for news
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, pred_length)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, price_seq, news_embed):
        batch_size = price_seq.size(0)

        # Embed price sequence
        price_emb = self.price_embedding(price_seq)
        price_emb = price_emb + self.pos_encoding

        # Project news embeddings
        news_emb = self.news_projection(news_embed)

        # Transformer encoding of price
        price_encoded = self.transformer(price_emb)

        # Cross-attention: price attends to news
        combined, _ = self.cross_attention(
            query=price_encoded,
            key=news_emb,
            value=news_emb
        )

        # Use last timestep for prediction
        out = combined[:, -1, :]

        # MLP head
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)

        return out

# ============================================
# Data Preparation
# ============================================
def prepare_data(data_dir, seq_length=30, pred_length=5, chromadb_dir="/tmp/chromadb"):
    """Prepare price and news data"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    ticker = "^GSPC"
    end_date = datetime.now()
    raw_data_path = data_path / 'raw_data.pt'

    # Check for existing data
    existing_data = None
    last_date = None

    if raw_data_path.exists():
        print("Loading existing raw data...")
        existing_data = torch.load(raw_data_path, map_location='cpu', weights_only=False)
        last_date = existing_data.get('last_date')
        print(f"Existing data: {existing_data['num_days']} days, last_date: {last_date}")

    # Determine download range
    if last_date:
        start_date = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)
        print(f"Incremental mode: downloading from {start_date.strftime('%Y-%m-%d')}")
    else:
        start_date = end_date - timedelta(days=3600)
        print("Initial mode: downloading 10 years of data")

    # 1. Download price data
    print(f"Downloading {ticker} price data...")
    new_df = yf.download(ticker, start=start_date, end=end_date, progress=True)

    if len(new_df) == 0 and existing_data:
        print("No new data available. Using existing data.")
        train_data = torch.load(data_path / 'train_data.pt', map_location='cpu', weights_only=False)
        test_data = torch.load(data_path / 'test_data.pt', map_location='cpu', weights_only=False)
        with open(data_path / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return train_data, test_data, scaler

    print(f"Downloaded {len(new_df)} new days of price data")

    # Merge with existing data if available
    if existing_data and len(new_df) > 0:
        existing_prices = existing_data['prices']
        new_prices = new_df['Close'].values.reshape(-1, 1)
        all_prices = np.vstack([existing_prices, new_prices])

        existing_dates = existing_data['dates']
        new_dates = new_df.index.strftime('%Y-%m-%d').tolist()
        all_dates = existing_dates + new_dates

        print(f"Merged: {len(existing_prices)} + {len(new_prices)} = {len(all_prices)} days")
    else:
        all_prices = new_df['Close'].values.reshape(-1, 1)
        all_dates = new_df.index.strftime('%Y-%m-%d').tolist()

    # Normalize prices
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(all_prices)

    with open(data_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 2. Collect news data
    print("Collecting news data...")
    news_collector = NewsCollector()

    if last_date:
        news_start = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)
    else:
        news_start = start_date

    all_news = []
    for symbol in ['SPY', 'AAPL', 'MSFT', 'GOOGL']:
        news = news_collector.get_news(
            symbol=symbol,
            from_date=news_start.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d')
        )
        all_news.extend(news)

    market_news = news_collector.get_market_news()
    all_news.extend(market_news)

    print(f"Collected {len(all_news)} new news articles")

    # Organize news by date
    news_by_date = {}
    for news_item in all_news:
        if 'datetime' in news_item:
            news_date = datetime.fromtimestamp(news_item['datetime']).strftime('%Y-%m-%d')
        else:
            continue
        if news_date not in news_by_date:
            news_by_date[news_date] = []
        news_by_date[news_date].append(news_item)

    # 3. Create news embeddings
    print("Creating news embeddings with FinBERT...")
    embedder = NewsEmbedder(device=str(device))

    # 4. Initialize ChromaDB
    print("Initializing ChromaDB...")
    vector_store = NewsVectorStore(persist_dir=chromadb_dir)

    if all_news:
        print(f"Storing {len(all_news)} news embeddings in ChromaDB...")
        batch_size = 100
        for i in range(0, len(all_news), batch_size):
            batch_news = all_news[i:i+batch_size]
            headlines = [n.get('headline', '') for n in batch_news]
            if headlines:
                batch_embeddings = embedder.embed(headlines)
                vector_store.add_news(batch_news, batch_embeddings)

    stats = vector_store.get_stats()
    print(f"ChromaDB stats: {stats['total_documents']} total documents")

    # Create embeddings for each trading day
    news_embeddings = []

    if existing_data and 'news_embeddings' in existing_data:
        existing_news_emb = existing_data['news_embeddings']
        news_embeddings = list(existing_news_emb)
        print(f"Loaded {len(news_embeddings)} existing news embeddings")

        new_dates_only = all_dates[len(existing_data['dates']):]
        for date in new_dates_only:
            if date in news_by_date:
                headlines = [n.get('headline', '') for n in news_by_date[date][:5]]
                combined = ' '.join(headlines) if headlines else ''
                if combined:
                    emb = embedder.embed([combined]).squeeze()
                else:
                    emb = torch.zeros(768)
            else:
                db_news = vector_store.get_by_date(date)
                if db_news and db_news['documents']:
                    combined = ' '.join(db_news['documents'][:5])
                    emb = embedder.embed([combined]).squeeze()
                else:
                    emb = torch.zeros(768)
            news_embeddings.append(emb)

        print(f"Created {len(new_dates_only)} new news embeddings")
    else:
        for date in all_dates:
            if date in news_by_date:
                headlines = [n.get('headline', '') for n in news_by_date[date][:5]]
                combined = ' '.join(headlines) if headlines else ''
                if combined:
                    emb = embedder.embed([combined]).squeeze()
                else:
                    emb = torch.zeros(768)
            else:
                emb = torch.zeros(768)
            news_embeddings.append(emb)

    news_embeddings = torch.stack(news_embeddings)
    print(f"Total news embeddings: {news_embeddings.shape}")

    # Save raw data
    torch.save({
        'prices': all_prices,
        'dates': all_dates,
        'news_embeddings': news_embeddings,
        'last_date': all_dates[-1],
        'num_days': len(all_dates),
        'updated_at': datetime.now().isoformat()
    }, raw_data_path)
    print(f"Saved raw data: {len(all_dates)} days, last_date: {all_dates[-1]}")

    # 5. Create sequences
    X_price, X_news, y = [], [], []

    for i in range(len(prices_scaled) - seq_length - pred_length):
        X_price.append(prices_scaled[i:i+seq_length])
        X_news.append(news_embeddings[i:i+seq_length])
        y.append(prices_scaled[i+seq_length:i+seq_length+pred_length].flatten())

    X_price = torch.FloatTensor(np.array(X_price))
    X_news = torch.stack(X_news)
    y = torch.FloatTensor(np.array(y))

    # Train/test split
    split = int(len(X_price) * 0.8)

    train_data = {
        'X_price': X_price[:split],
        'X_news': X_news[:split],
        'y': y[:split]
    }

    test_data = {
        'X_price': X_price[split:],
        'X_news': X_news[split:],
        'y': y[split:]
    }

    torch.save(train_data, data_path / 'train_data.pt')
    torch.save(test_data, data_path / 'test_data.pt')

    # Save last sequence for inference
    last_price = prices_scaled[-seq_length:]
    last_news = news_embeddings[-seq_length:]
    torch.save({
        'price': torch.FloatTensor(last_price),
        'news': last_news
    }, data_path / 'last_sequence.pt')

    print(f"Training samples: {len(train_data['X_price'])}")
    print(f"Test samples: {len(test_data['X_price'])}")

    return train_data, test_data, scaler

# ============================================
# Checkpoint Functions (Single GPU version)
# ============================================
def save_checkpoint(model, optimizer, scheduler, epoch, loss, best_loss, checkpoint_dir, is_best=False):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  # No .module for single GPU
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'best_loss': best_loss,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, checkpoint_dir / 'checkpoint_latest.pt')
    print(f"Checkpoint saved: epoch {epoch+1}, loss {loss:.6f}")

    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'checkpoint_best.pt')
        print(f"Best model saved: loss {loss:.6f}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir, device):
    checkpoint_path = Path(checkpoint_dir) / 'checkpoint_latest.pt'

    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])  # No .module
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint.get('best_loss', float('inf'))
    return 0, float('inf')

# ============================================
# Evaluation
# ============================================
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for price, news, target in test_loader:
            price = price.to(device)
            news = news.to(device)
            target = target.to(device)

            output = model(price, news)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(test_loader)

# ============================================
# Export for Triton
# ============================================
def export_model_for_triton(model, export_dir, seq_length=30, pred_length=5, news_dim=768):
    """Export model to ONNX format for Triton"""
    export_dir = Path(export_dir)
    triton_model_dir = export_dir / "stock_predictor" / "1"
    triton_model_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model_device = next(model.parameters()).device

    dummy_price = torch.randn(1, seq_length, 1).to(model_device)
    dummy_news = torch.randn(1, seq_length, news_dim).to(model_device)

    onnx_path = triton_model_dir / 'model.onnx'

    try:
        torch.onnx.export(
            model,
            (dummy_price, dummy_news),
            str(onnx_path),
            export_params=True,
            opset_version=17,
            input_names=['price_input', 'news_input'],
            output_names=['output'],
            dynamic_axes={
                'price_input': {0: 'batch'},
                'news_input': {0: 'batch'},
                'output': {0: 'batch'}
            },
            dynamo=False  # Use legacy export to avoid onnxscript dependency
        )
        print(f"ONNX model saved: {onnx_path}")
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
        print("Skipping ONNX export. Install onnxscript for full export support: pip install onnxscript")
        # Save PyTorch model instead
        torch.save(model.state_dict(), triton_model_dir / 'model.pt')
        print(f"PyTorch model saved: {triton_model_dir / 'model.pt'}")

    config_content = f"""name: "stock_predictor"
platform: "onnxruntime_onnx"
max_batch_size: 64
input [
  {{
    name: "price_input"
    data_type: TYPE_FP32
    dims: [ {seq_length}, 1 ]
  }},
  {{
    name: "news_input"
    data_type: TYPE_FP32
    dims: [ {seq_length}, {news_dim} ]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ {pred_length} ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
dynamic_batching {{
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}}
"""
    config_path = export_dir / "stock_predictor" / "config.pbtxt"
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"Triton config saved: {config_path}")

    metadata = {
        'model_name': 'stock_predictor',
        'model_type': 'Transformer + FinBERT',
        'task': 'stock_price_prediction',
        'ticker': '^GSPC',
        'input': {
            'price_shape': [1, seq_length, 1],
            'news_shape': [1, seq_length, news_dim]
        },
        'output_shape': [1, pred_length],
        'seq_length': seq_length,
        'pred_length': pred_length,
        'news_embedding': 'FinBERT (ProsusAI/finbert)',
        'export_timestamp': datetime.now().isoformat()
    }
    with open(export_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

# ============================================
# Main Training Function
# ============================================
def train():
    """Main training function for Jupyter"""
    print("=" * 60)
    print("S&P 500 Stock Prediction - Transformer + News")
    print("Jupyter Notebook Version")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Configuration: {vars(args)}")

    # Prepare data
    train_data, test_data, scaler = prepare_data(
        args.data_dir, args.seq_length, args.pred_length, args.chromadb_dir
    )

    # Create datasets
    train_dataset = StockNewsDataset(
        train_data['X_price'],
        train_data['X_news'],
        train_data['y']
    )
    test_dataset = StockNewsDataset(
        test_data['X_price'],
        test_data['X_news'],
        test_data['y']
    )

    # Data loaders (no DistributedSampler)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2
    )

    # Model (no DDP)
    model = StockNewsTransformer(
        price_dim=1,
        news_dim=768,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        pred_length=args.pred_length
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Load checkpoint
    start_epoch, best_loss = 0, float('inf')
    if args.resume:
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, args.checkpoint_dir, device)
        print(f"Resumed from epoch {start_epoch}, best_loss: {best_loss:.6f}")

    # Training loop
    train_losses = []
    test_losses = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (price, news, target) in enumerate(train_loader):
            price = price.to(device)
            news = news.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(price, news)
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluation
        test_loss = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train: {avg_loss:.6f} Test: {test_loss:.6f} LR: {current_lr:.6f}")

        is_best = test_loss < best_loss
        if is_best:
            best_loss = test_loss
        save_checkpoint(model, optimizer, scheduler, epoch, test_loss, best_loss,
                       args.checkpoint_dir, is_best)

    # Export
    print("\n" + "=" * 60)
    print("Exporting model for Triton...")
    print("=" * 60)

    best_ckpt = Path(args.checkpoint_dir) / 'checkpoint_best.pt'
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best model: loss {ckpt['best_loss']:.6f}")

    export_model_for_triton(
        model, args.export_dir,
        args.seq_length, args.pred_length
    )

    # Copy artifacts
    import shutil
    data_path = Path(args.data_dir)
    shutil.copy(data_path / 'scaler.pkl', Path(args.export_dir) / 'scaler.pkl')
    shutil.copy(data_path / 'last_sequence.pt', Path(args.export_dir) / 'last_sequence.pt')

    # Summary
    summary = {
        'task': 'S&P 500 Stock Prediction',
        'model': 'Transformer + FinBERT News',
        'ticker': '^GSPC',
        'best_loss': best_loss,
        'total_epochs': args.epochs,
        'device': str(device),
        'seq_length': args.seq_length,
        'pred_length': args.pred_length,
        'completed_at': datetime.now().isoformat()
    }
    with open(Path(args.export_dir) / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining completed! Best loss: {best_loss:.6f}")
    print(f"Model: Transformer + FinBERT News Embeddings")
    print(f"Exported to: {args.export_dir}/stock_predictor/")

    return model, train_losses, test_losses, scaler

# ============================================
# Run Training (Jupyter에서 실행)
# ============================================
if __name__ == "__main__":
    model, train_losses, test_losses, scaler = train()
