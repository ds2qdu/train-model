# ============================================
# S&P 500 Stock Prediction - Transformer + News
# Distributed Training with Incremental Learning
# ============================================

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import json
import pickle
import requests
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Setup
# ============================================
def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

# ============================================
# GPU Monitoring
# ============================================
def get_gpu_info():
    """Get GPU utilization and memory info"""
    if not torch.cuda.is_available():
        return "GPU: N/A"

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                util, mem_used, mem_total = line.split(', ')
                gpu_info.append(f"GPU{i}: {util}% | {mem_used}/{mem_total}MB")
            return ' | '.join(gpu_info)
    except:
        pass

    # Fallback to PyTorch info
    device = torch.cuda.current_device()
    mem_used = torch.cuda.memory_allocated(device) / 1024**3
    mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"GPU{device}: Mem {mem_used:.1f}/{mem_total:.1f}GB"

def print_gpu_status(rank, epoch=None, batch_idx=None):
    """Print GPU status with rank info"""
    gpu_info = get_gpu_info()
    if epoch is not None and batch_idx is not None:
        print(f"[Rank {rank}] Epoch {epoch+1} Batch {batch_idx} | {gpu_info}")
    else:
        print(f"[Rank {rank}] {gpu_info}")

# ============================================
# News Data Collection (Finnhub)
# ============================================
class NewsCollector:
    def __init__(self, api_key=None):
        # Finnhub free API key (can be overridden via env)
        self.api_key = api_key or os.environ.get('FINNHUB_API_KEY', 'demo')
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
        self.embedding_dim = 768  # FinBERT hidden size

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
                    text[:512],  # Truncate long texts
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                ).to(self.device)

                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(cls_embedding.cpu())

        return torch.stack(embeddings) if embeddings else torch.zeros(1, self.embedding_dim)

    def embed_daily_news(self, news_by_date):
        """Embed news aggregated by date"""
        daily_embeddings = {}

        for date, news_list in news_by_date.items():
            if news_list:
                # Concatenate headlines for the day
                headlines = [n.get('headline', '') for n in news_list[:10]]  # Max 10 per day
                combined_text = ' '.join(headlines)
                embedding = self.embed([combined_text])
                daily_embeddings[date] = embedding.squeeze()
            else:
                daily_embeddings[date] = torch.zeros(self.embedding_dim)

        return daily_embeddings

# ============================================
# ChromaDB Vector Store
# ============================================
class NewsVectorStore:
    def __init__(self, persist_dir="/mnt/storage/chromadb"):
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
        """
        Add news items with embeddings to ChromaDB

        Args:
            news_items: list of news dictionaries with keys: headline, source, datetime, url
            embeddings: torch.Tensor of shape (n, 768)
        """
        if len(news_items) == 0:
            return

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embedding_list = []
        seen_ids = set()  # 배치 내 중복 체크용

        for i, news in enumerate(news_items):
            # Create unique ID based on datetime and URL/headline hash
            news_datetime = news.get('datetime', 0)
            headline = news.get('headline', '')
            url = news.get('url', '')
            # URL이 있으면 URL hash 사용, 없으면 headline hash 사용
            unique_key = url if url else headline
            news_id = f"{news_datetime}_{hash(unique_key) % 10000000}"

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

    def search(self, query_embedding, n_results=10, date_filter=None):
        """
        Search similar news by embedding

        Args:
            query_embedding: torch.Tensor of shape (768,)
            n_results: number of results to return
            date_filter: optional date string to filter (YYYY-MM-DD)
        """
        query = query_embedding.tolist() if torch.is_tensor(query_embedding) else query_embedding

        where_filter = None
        if date_filter:
            where_filter = {"date": date_filter}

        results = self.collection.query(
            query_embeddings=[query],
            n_results=n_results,
            where=where_filter
        )

        return results

    def get_by_date(self, date):
        """Get all news for a specific date"""
        results = self.collection.get(
            where={"date": date},
            include=["embeddings", "documents", "metadatas"]
        )
        return results

    def get_stats(self):
        """Get collection statistics"""
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
        """
        Args:
            price_seq: (batch, seq_len, 1) - price time series
            news_embed: (batch, seq_len, 768) - news embeddings per day
        Returns:
            (batch, pred_length) - predictions
        """
        batch_size = price_seq.size(0)

        # Embed price sequence
        price_emb = self.price_embedding(price_seq)  # (batch, seq, d_model)
        price_emb = price_emb + self.pos_encoding

        # Project news embeddings
        news_emb = self.news_projection(news_embed)  # (batch, seq, d_model)

        # Transformer encoding of price
        price_encoded = self.transformer(price_emb)  # (batch, seq, d_model)

        # Cross-attention: price attends to news
        combined, _ = self.cross_attention(
            query=price_encoded,
            key=news_emb,
            value=news_emb
        )

        # Use last timestep for prediction
        out = combined[:, -1, :]  # (batch, d_model)

        # MLP head
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)

        return out

# ============================================
# Data Preparation (Incremental)
# ============================================
def prepare_data(data_dir, seq_length=30, pred_length=5, rank=0, chromadb_dir="/mnt/storage/chromadb", skip_news_fetch=False):
    """
    Prepare price and news data with incremental loading.
    - If raw_data.pt exists: load existing data and append new data only
    - If not: download full 10-year data (initial training)
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    ticker = "^GSPC"

    # Use US Eastern timezone for market dates
    us_eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(us_eastern)
    end_date = now_eastern.replace(tzinfo=None)  # yfinance expects naive datetime

    print(f"[Rank {rank}] Current US Eastern time: {now_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    raw_data_path = data_path / 'raw_data.pt'

    # Check for existing data
    existing_data = None
    last_date = None

    if raw_data_path.exists():
        print(f"[Rank {rank}] Loading existing raw data...")
        existing_data = torch.load(raw_data_path, map_location='cpu', weights_only=False)
        last_date = existing_data.get('last_date')
        print(f"[Rank {rank}] Existing data: {existing_data['num_days']} days, last_date: {last_date}")

    # Determine download range
    if last_date:
        # Incremental: download from last_date (inclusive to catch any missed data)
        start_date = datetime.strptime(last_date, '%Y-%m-%d')

        # If start_date is today or in the future, no new data to download
        if start_date.date() >= end_date.date():
            print(f"[Rank {rank}] Data is up to date. No new download needed.")
            if existing_data:
                train_data = torch.load(data_path / 'train_data.pt', map_location='cpu', weights_only=False)
                test_data = torch.load(data_path / 'test_data.pt', map_location='cpu', weights_only=False)
                with open(data_path / 'scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                return train_data, test_data, scaler

        print(f"[Rank {rank}] Incremental mode: downloading from {start_date.strftime('%Y-%m-%d')}")
    else:
        # Initial: download full 10 years
        start_date = end_date - timedelta(days=3600)
        print(f"[Rank {rank}] Initial mode: downloading 10 years of data")

    # 1. Download price data
    print(f"[Rank {rank}] Downloading {ticker} price data...")
    print(f"[Rank {rank}] Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    new_df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if len(new_df) == 0:
        if existing_data:
            print(f"[Rank {rank}] No new data available (market closed or weekend). Using existing data.")
        # Load existing processed data
        train_data = torch.load(data_path / 'train_data.pt', map_location='cpu', weights_only=False)
        test_data = torch.load(data_path / 'test_data.pt', map_location='cpu', weights_only=False)
        with open(data_path / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return train_data, test_data, scaler

    print(f"[Rank {rank}] Downloaded {len(new_df)} new days of price data")

    # Merge with existing data if available
    if existing_data and len(new_df) > 0:
        # Combine prices
        existing_prices = existing_data['prices']
        new_prices = new_df['Close'].values.reshape(-1, 1)
        all_prices = np.vstack([existing_prices, new_prices])

        # Combine dates
        existing_dates = existing_data['dates']
        new_dates = new_df.index.strftime('%Y-%m-%d').tolist()
        all_dates = existing_dates + new_dates

        print(f"[Rank {rank}] Merged: {len(existing_prices)} + {len(new_prices)} = {len(all_prices)} days")
    else:
        all_prices = new_df['Close'].values.reshape(-1, 1)
        all_dates = new_df.index.strftime('%Y-%m-%d').tolist()

    # Normalize prices (refit scaler on all data)
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(all_prices)

    # Save scaler
    with open(data_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 2. Collect news data (only for new dates)
    all_news = []

    if skip_news_fetch:
        print(f"[Rank {rank}] Skipping news fetch (--skip-news-fetch enabled)")
    else:
        print(f"[Rank {rank}] Collecting news data...")
        news_collector = NewsCollector()

        # Determine news collection range
        if last_date:
            news_start = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)
        else:
            news_start = start_date

        for symbol in ['SPY', 'AAPL', 'MSFT', 'GOOGL']:
            news = news_collector.get_news(
                symbol=symbol,
                from_date=news_start.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )
            all_news.extend(news)

        market_news = news_collector.get_market_news()
        all_news.extend(market_news)

        print(f"[Rank {rank}] Collected {len(all_news)} new news articles")

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
    print(f"[Rank {rank}] Creating news embeddings with FinBERT...")
    embedder = NewsEmbedder(device='cuda' if torch.cuda.is_available() else 'cpu')

    # 4. Initialize ChromaDB and store new news embeddings
    print(f"[Rank {rank}] Initializing ChromaDB...")
    vector_store = NewsVectorStore(persist_dir=chromadb_dir)

    # Store individual news embeddings in ChromaDB
    if all_news:
        print(f"[Rank {rank}] Storing {len(all_news)} news embeddings in ChromaDB...")
        batch_size = 100
        for i in range(0, len(all_news), batch_size):
            batch_news = all_news[i:i+batch_size]
            headlines = [n.get('headline', '') for n in batch_news]
            if headlines:
                batch_embeddings = embedder.embed(headlines)
                vector_store.add_news(batch_news, batch_embeddings)

    stats = vector_store.get_stats()
    print(f"[Rank {rank}] ChromaDB stats: {stats['total_documents']} total documents")

    # Create embeddings for each trading day
    # For existing dates, load from ChromaDB; for new dates, create embeddings
    news_embeddings = []

    if existing_data and 'news_embeddings' in existing_data:
        # Use existing embeddings for old dates
        existing_news_emb = existing_data['news_embeddings']
        news_embeddings = list(existing_news_emb)
        print(f"[Rank {rank}] Loaded {len(news_embeddings)} existing news embeddings")

        # Create embeddings only for new dates
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
                # Try to get from ChromaDB
                db_news = vector_store.get_by_date(date)
                if db_news and db_news['documents']:
                    combined = ' '.join(db_news['documents'][:5])
                    emb = embedder.embed([combined]).squeeze()
                else:
                    emb = torch.zeros(768)
            news_embeddings.append(emb)

        print(f"[Rank {rank}] Created {len(new_dates_only)} new news embeddings")
    else:
        # Initial: create all embeddings
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
    print(f"[Rank {rank}] Total news embeddings: {news_embeddings.shape}")

    # Save raw data for future incremental loading
    torch.save({
        'prices': all_prices,
        'dates': all_dates,
        'news_embeddings': news_embeddings,
        'last_date': all_dates[-1],
        'num_days': len(all_dates),
        'updated_at': datetime.now().isoformat()
    }, raw_data_path)
    print(f"[Rank {rank}] Saved raw data: {len(all_dates)} days, last_date: {all_dates[-1]}")

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

    # Save for other ranks
    torch.save(train_data, data_path / 'train_data.pt')
    torch.save(test_data, data_path / 'test_data.pt')

    # Save last sequence for inference
    last_price = prices_scaled[-seq_length:]
    last_news = news_embeddings[-seq_length:]
    torch.save({
        'price': torch.FloatTensor(last_price),
        'news': last_news
    }, data_path / 'last_sequence.pt')

    print(f"[Rank {rank}] Training samples: {len(train_data['X_price'])}")
    print(f"[Rank {rank}] Test samples: {len(test_data['X_price'])}")

    return train_data, test_data, scaler

# ============================================
# Checkpoint Functions
# ============================================
def save_checkpoint(model, optimizer, scheduler, epoch, loss, best_loss, checkpoint_dir, is_best=False):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
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
        model.module.load_state_dict(checkpoint['model_state_dict'])
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
    device = next(model.parameters()).device

    # Dummy inputs
    dummy_price = torch.randn(1, seq_length, 1).to(device)
    dummy_news = torch.randn(1, seq_length, news_dim).to(device)

    # ONNX export
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
            dynamo=False  # Use legacy export to avoid onnxscript issues
        )
        print(f"ONNX model saved: {onnx_path}")
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
        # Save PyTorch model as fallback
        torch.save(model.state_dict(), triton_model_dir / 'model.pt')
        print(f"PyTorch model saved: {triton_model_dir / 'model.pt'}")

    # Triton config
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

    # Metadata
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
# Main Training
# ============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seq-length', type=int, default=30)
    parser.add_argument('--pred-length', type=int, default=5)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--data-dir', type=str, default='/mnt/storage/data')
    parser.add_argument('--checkpoint-dir', type=str, default='/mnt/storage/checkpoints')
    parser.add_argument('--export-dir', type=str, default='/mnt/storage/models')
    parser.add_argument('--chromadb-dir', type=str, default='/mnt/storage/chromadb')
    parser.add_argument('--tensorboard-dir', type=str, default='/mnt/tensorboard')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint (incremental learning)')
    parser.add_argument('--skip-news-fetch', action='store_true', help='Skip fetching new news from API (use existing ChromaDB data)')
    args = parser.parse_args()

    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')

    # TensorBoard writer (rank 0 only)
    writer = None
    if rank == 0:
        # 타임스탬프 서브디렉토리 사용 (노트북과 동일 패턴)
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tb_log_dir = str(Path(args.tensorboard_dir) / run_name)
        Path(tb_log_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)

        # 즉시 초기 이벤트 기록 + flush (파일 생성 보장)
        writer.add_scalar('Status/init', 1.0, 0)
        writer.flush()
        print(f"TensorBoard event file created at: {tb_log_dir}")

        print("=" * 60)
        print("S&P 500 Stock Prediction - Transformer + News")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Incremental learning: {args.resume}")
        print(f"TensorBoard log dir: {tb_log_dir}")
        print(f"Arguments: {args}")

    # Prepare data (rank 0 only)
    data_path = Path(args.data_dir)
    if rank == 0:
        train_data, test_data, scaler = prepare_data(
            args.data_dir, args.seq_length, args.pred_length, rank, args.chromadb_dir,
            skip_news_fetch=args.skip_news_fetch
        )

    dist.barrier()

    # Load data on all ranks
    train_data = torch.load(data_path / 'train_data.pt', map_location='cpu', weights_only=False)
    test_data = torch.load(data_path / 'test_data.pt', map_location='cpu', weights_only=False)

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

    # Samplers and loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4
    )

    # Model
    model = StockNewsTransformer(
        price_dim=1,
        news_dim=768,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        pred_length=args.pred_length
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Load checkpoint for incremental learning
    start_epoch, best_loss = 0, float('inf')
    if args.resume:
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, args.checkpoint_dir, device)
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}, best_loss: {best_loss:.6f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            # TensorBoard: batch loss + GPU memory
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

                # GPU 메모리 (매 50 배치마다)
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    gpu_mem_used = torch.cuda.memory_allocated(device) / 1024**3
                    gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    writer.add_scalar('System/gpu_memory_allocated_GB', gpu_mem_used, global_step)
                    writer.add_scalar('System/gpu_memory_reserved_GB', gpu_mem_reserved, global_step)

            # GPU 상태 출력 (매 50 배치마다)
            if batch_idx % 50 == 0:
                print_gpu_status(rank, epoch, batch_idx)
                print(f"[Rank {rank}] Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # 에폭 종료 시 GPU 상태 출력
        print_gpu_status(rank)
        print(f"[Rank {rank}] Epoch [{epoch+1}/{args.epochs}] completed. Avg Loss: {avg_loss:.6f}")

        # Evaluation (rank 0)
        if rank == 0:
            test_loss = evaluate(model.module, test_loader, criterion, device)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"Train: {avg_loss:.6f} Test: {test_loss:.6f} LR: {current_lr:.6f}")

            # TensorBoard: epoch metrics
            writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)
            writer.add_scalars('Loss/comparison', {
                'train': avg_loss,
                'test': test_loss,
            }, epoch)

            # TensorBoard: 파라미터 & 그래디언트 히스토그램 (매 5 에폭)
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                for name, param in model.module.named_parameters():
                    writer.add_histogram(f'Parameters/{name}', param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

            # TensorBoard: gradient norm
            total_norm = 0.0
            for p in model.module.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            writer.add_scalar('Training/gradient_norm', total_norm, epoch)

            # TensorBoard: 예측 vs 실제 샘플 비교 (매 5 에폭)
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    sample_price, sample_news, sample_target = next(iter(test_loader))
                    sample_price = sample_price[:4].to(device)
                    sample_news = sample_news[:4].to(device)
                    sample_target = sample_target[:4].to(device)

                    sample_output = model.module(sample_price, sample_news)

                    for i in range(min(4, sample_output.size(0))):
                        for day in range(sample_output.size(1)):
                            writer.add_scalar(f'Predictions/sample_{i}/pred_day_{day}', sample_output[i][day].item(), epoch)
                            writer.add_scalar(f'Predictions/sample_{i}/target_day_{day}', sample_target[i][day].item(), epoch)
                model.train()

            is_best = test_loss < best_loss
            if is_best:
                best_loss = test_loss
                writer.add_scalar('Loss/best', best_loss, epoch)

            # TensorBoard: flush to disk every epoch
            writer.flush()

            save_checkpoint(model, optimizer, scheduler, epoch, test_loss, best_loss,
                          args.checkpoint_dir, is_best)

        dist.barrier()

    # Export (rank 0)
    if rank == 0:
        print("\n" + "=" * 60)
        print("Exporting model for Triton...")
        print("=" * 60)

        best_ckpt = Path(args.checkpoint_dir) / 'checkpoint_best.pt'
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.module.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded best model: loss {ckpt['best_loss']:.6f}")

        export_model_for_triton(
            model.module, args.export_dir,
            args.seq_length, args.pred_length
        )

        # Copy artifacts
        import shutil
        shutil.copy(data_path / 'scaler.pkl', Path(args.export_dir) / 'scaler.pkl')
        shutil.copy(data_path / 'last_sequence.pt', Path(args.export_dir) / 'last_sequence.pt')

        # Summary
        summary = {
            'task': 'S&P 500 Stock Prediction',
            'model': 'Transformer + FinBERT News',
            'ticker': '^GSPC',
            'best_loss': best_loss,
            'total_epochs': args.epochs,
            'world_size': world_size,
            'incremental_learning': args.resume,
            'seq_length': args.seq_length,
            'pred_length': args.pred_length,
            'completed_at': datetime.now().isoformat()
        }
        with open(Path(args.export_dir) / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n Training completed! Best loss: {best_loss:.6f}")
        print(f" Model: Transformer + FinBERT News Embeddings")
        print(f" Exported to: {args.export_dir}/stock_predictor/")

    # TensorBoard: 최종 요약 텍스트 + close
    if writer is not None:
        summary_text = (
            f"# Training Summary\n\n"
            f"- Best Loss: {best_loss:.6f}\n"
            f"- Total Epochs: {args.epochs}\n"
            f"- World Size: {world_size}\n"
            f"- Model: Transformer + FinBERT News\n"
            f"- Sequence Length: {args.seq_length}\n"
            f"- Prediction Length: {args.pred_length}\n"
            f"- Incremental Learning: {args.resume}\n"
        )
        writer.add_text('Summary', summary_text, 0)
        writer.flush()
        writer.close()

    cleanup()

if __name__ == "__main__":
    main()
