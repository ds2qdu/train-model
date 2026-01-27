# ============================================
# Stock Prediction RAG Chatbot
# LangChain + ChromaDB + Local LLM + Triton
# Supports: Ollama, vLLM, OpenAI
# ============================================

import os
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage

# Configuration
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", "/mnt/storage/chromadb")
TRITON_URL = os.environ.get("TRITON_URL", "http://stock-predictor-triton:8000")

# LLM Configuration
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")  # ollama, vllm, openai
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2")  # Model name
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
VLLM_URL = os.environ.get("VLLM_URL", "http://vllm:8000")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ============================================
# Pydantic Models
# ============================================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    prediction: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    prices: List[float]  # Last 30 days of prices
    news_query: Optional[str] = None

class PredictionResponse(BaseModel):
    predictions: List[float]  # Next 5 days predictions
    confidence: float
    related_news: List[Dict[str, Any]] = []

# ============================================
# LLM Provider Abstraction
# ============================================
class LLMProvider:
    """Abstract LLM provider supporting multiple backends"""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.llm = None

        if provider == "ollama":
            self._init_ollama()
        elif provider == "vllm":
            self._init_vllm()
        elif provider == "openai":
            self._init_openai()
        else:
            print(f"Unknown LLM provider: {provider}. Using Ollama as default.")
            self._init_ollama()

    def _init_ollama(self):
        """Initialize Ollama LLM"""
        try:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=self.model,
                base_url=OLLAMA_URL,
                temperature=0.7
            )
            print(f"Initialized Ollama LLM: {self.model} at {OLLAMA_URL}")
        except ImportError:
            print("langchain-ollama not installed. Install with: pip install langchain-ollama")
            self.llm = None

    def _init_vllm(self):
        """Initialize vLLM (OpenAI-compatible API)"""
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.model,
                base_url=f"{VLLM_URL}/v1",
                api_key="not-needed",  # vLLM doesn't require API key
                temperature=0.7
            )
            print(f"Initialized vLLM: {self.model} at {VLLM_URL}")
        except ImportError:
            print("langchain-openai not installed. Install with: pip install langchain-openai")
            self.llm = None

    def _init_openai(self):
        """Initialize OpenAI LLM"""
        if not OPENAI_API_KEY:
            print("OPENAI_API_KEY not set")
            self.llm = None
            return

        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=OPENAI_API_KEY,
                temperature=0.7
            )
            print(f"Initialized OpenAI LLM: {self.model}")
        except ImportError:
            print("langchain-openai not installed")
            self.llm = None

    def invoke(self, messages: List) -> str:
        """Send messages to LLM and get response"""
        if not self.llm:
            return "LLM is not available. Please check configuration."

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"LLM error: {str(e)}"

    def is_available(self) -> bool:
        """Check if LLM is available"""
        return self.llm is not None

# ============================================
# ChromaDB News Retriever
# ============================================
class NewsRetriever:
    def __init__(self, chromadb_path: str):
        self.client = chromadb.PersistentClient(path=chromadb_path)
        try:
            self.collection = self.client.get_collection("stock_news")
            print(f"Loaded ChromaDB collection: {self.collection.count()} documents")
        except Exception as e:
            print(f"Warning: Could not load ChromaDB collection: {e}")
            self.collection = None

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant news using semantic similarity"""
        if not self.collection:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            news_list = []
            for i in range(len(results['ids'][0])):
                news_list.append({
                    "headline": results['documents'][0][i],
                    "date": results['metadatas'][0][i].get('date', 'N/A'),
                    "source": results['metadatas'][0][i].get('source', 'N/A'),
                    "symbol": results['metadatas'][0][i].get('symbol', 'N/A'),
                    "relevance": 1 - results['distances'][0][i]
                })
            return news_list
        except Exception as e:
            print(f"News search error: {e}")
            return []

    def get_recent_news(self, days: int = 7, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent news from the last N days"""
        if not self.collection:
            return []

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            all_data = self.collection.get(
                include=["documents", "metadatas"],
                limit=1000
            )

            news_list = []
            for i, meta in enumerate(all_data['metadatas']):
                news_date = meta.get('date', '')
                if news_date >= start_date.strftime('%Y-%m-%d'):
                    news_list.append({
                        "headline": all_data['documents'][i],
                        "date": news_date,
                        "source": meta.get('source', 'N/A'),
                        "symbol": meta.get('symbol', 'N/A')
                    })

            news_list.sort(key=lambda x: x['date'], reverse=True)
            return news_list[:limit]
        except Exception as e:
            print(f"Recent news error: {e}")
            return []

# ============================================
# Triton Client
# ============================================
class TritonClient:
    def __init__(self, triton_url: str):
        self.triton_url = triton_url.rstrip('/')
        self.model_name = "stock_predictor"

    def is_healthy(self) -> bool:
        """Check if Triton server is healthy"""
        try:
            response = requests.get(f"{self.triton_url}/v2/health/ready", timeout=5)
            return response.status_code == 200
        except:
            return False

    def predict(self, price_input: np.ndarray, news_input: np.ndarray) -> Optional[np.ndarray]:
        """Make prediction using Triton inference server"""
        try:
            inference_request = {
                "inputs": [
                    {
                        "name": "price_input",
                        "shape": list(price_input.shape),
                        "datatype": "FP32",
                        "data": price_input.flatten().tolist()
                    },
                    {
                        "name": "news_input",
                        "shape": list(news_input.shape),
                        "datatype": "FP32",
                        "data": news_input.flatten().tolist()
                    }
                ]
            }

            response = requests.post(
                f"{self.triton_url}/v2/models/{self.model_name}/infer",
                json=inference_request,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                output_data = result['outputs'][0]['data']
                return np.array(output_data).reshape(-1, 5)
            else:
                print(f"Triton inference error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Triton client error: {e}")
            return None

# ============================================
# RAG Chatbot
# ============================================
class StockChatbot:
    def __init__(self):
        self.news_retriever = NewsRetriever(CHROMADB_PATH)
        self.triton_client = TritonClient(TRITON_URL)

        # Initialize LLM (Local or Cloud)
        self.llm_provider = LLMProvider(LLM_PROVIDER, LLM_MODEL)

        # Simple conversation history per session (list of messages)
        self.histories: Dict[str, List] = {}

        # System prompt
        self.system_prompt = """You are a helpful stock market analyst assistant. You have access to:
1. Recent financial news from ChromaDB vector database
2. S&P 500 price prediction model (Transformer-based)

When answering questions:
- Provide analysis based on the retrieved news context
- If price predictions are available, explain them in context
- Be clear about uncertainties and risks
- Never give specific investment advice or guarantees
- Always mention this is for informational purposes only

Current date: {current_date}
"""

    def get_history(self, session_id: str) -> List:
        """Get or create conversation history for a session"""
        if session_id not in self.histories:
            self.histories[session_id] = []
        return self.histories[session_id]

    def chat(self, message: str, session_id: str = "default") -> ChatResponse:
        """Process a chat message and return response"""
        if not self.llm_provider.is_available():
            return ChatResponse(
                response=f"LLM is not configured. Provider: {LLM_PROVIDER}, Model: {LLM_MODEL}",
                sources=[]
            )

        # 1. Search for relevant news
        relevant_news = self.news_retriever.search(message, n_results=5)

        # 2. Build context from news
        news_context = ""
        if relevant_news:
            news_context = "\n\nRelevant News:\n"
            for i, news in enumerate(relevant_news, 1):
                news_context += f"{i}. [{news['date']}] {news['headline']} (Source: {news['source']})\n"

        # 3. Get conversation history
        history = self.get_history(session_id)

        # 4. Build messages
        system_message = self.system_prompt.format(
            current_date=datetime.now().strftime('%Y-%m-%d')
        )

        messages = [
            SystemMessage(content=system_message + news_context),
            *history[-10:],  # Keep last 10 messages to avoid context overflow
            HumanMessage(content=message)
        ]

        # 5. Get LLM response
        assistant_response = self.llm_provider.invoke(messages)

        # Save to history
        from langchain_core.messages import AIMessage
        history.append(HumanMessage(content=message))
        history.append(AIMessage(content=assistant_response))

        return ChatResponse(
            response=assistant_response,
            sources=[{"headline": n["headline"], "date": n["date"], "source": n["source"]}
                    for n in relevant_news]
        )

# ============================================
# FastAPI Application
# ============================================
app = FastAPI(
    title="Stock Prediction Chatbot API",
    description="RAG-based chatbot with Local LLM support (Ollama/vLLM)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = StockChatbot()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "triton_available": chatbot.triton_client.is_healthy(),
        "chromadb_documents": chatbot.news_retriever.collection.count() if chatbot.news_retriever.collection else 0,
        "llm_available": chatbot.llm_provider.is_available(),
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for conversational interaction"""
    return chatbot.chat(request.message, request.session_id)

@app.get("/news/recent")
async def get_recent_news(days: int = 7, limit: int = 20):
    """Get recent news from ChromaDB"""
    news = chatbot.news_retriever.get_recent_news(days=days, limit=limit)
    return {"news": news, "count": len(news)}

@app.get("/news/search")
async def search_news(query: str, n_results: int = 10):
    """Search news by semantic similarity"""
    news = chatbot.news_retriever.search(query, n_results=n_results)
    return {"news": news, "count": len(news)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make price prediction using Triton inference server"""
    if not chatbot.triton_client.is_healthy():
        raise HTTPException(status_code=503, detail="Triton server not available")

    if len(request.prices) != 30:
        raise HTTPException(status_code=400, detail="Exactly 30 price values required")

    # Normalize prices
    prices = np.array(request.prices)
    min_price = prices.min()
    max_price = prices.max()
    normalized_prices = (prices - min_price) / (max_price - min_price + 1e-8)

    # Prepare input
    price_input = normalized_prices.reshape(1, 30, 1).astype(np.float32)
    news_input = np.zeros((1, 30, 768), dtype=np.float32)

    # Get prediction
    predictions = chatbot.triton_client.predict(price_input, news_input)

    if predictions is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    # Denormalize predictions
    denormalized = predictions[0] * (max_price - min_price) + min_price

    # Get related news
    related_news = []
    if request.news_query:
        related_news = chatbot.news_retriever.search(request.news_query, n_results=5)

    return PredictionResponse(
        predictions=denormalized.tolist(),
        confidence=0.85,
        related_news=related_news
    )

@app.get("/model/status")
async def model_status():
    """Get Triton model status"""
    try:
        response = requests.get(f"{TRITON_URL}/v2/models/stock_predictor", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ============================================
# Main
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
