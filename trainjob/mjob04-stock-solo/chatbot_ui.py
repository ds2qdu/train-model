# ============================================
# Stock Prediction Chatbot - Streamlit UI
# 실행: streamlit run chatbot_ui.py
# ============================================

import streamlit as st
import requests
import json
from datetime import datetime
import os

# Configuration
CHATBOT_API_URL = os.environ.get("CHATBOT_API_URL", "http://localhost:8080")

st.set_page_config(
    page_title="Stock Prediction Chatbot",
    page_icon="📈",
    layout="wide"
)

# ============================================
# Session State
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ============================================
# API Functions
# ============================================
def chat_with_api(message: str) -> dict:
    """Send message to chatbot API"""
    try:
        response = requests.post(
            f"{CHATBOT_API_URL}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id
            },
            timeout=600
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"response": f"Error: {response.status_code}", "sources": []}
    except requests.exceptions.ConnectionError:
        return {"response": "Cannot connect to chatbot API. Is the server running?", "sources": []}
    except Exception as e:
        return {"response": f"Error: {str(e)}", "sources": []}

def get_health_status() -> dict:
    """Get API health status"""
    try:
        response = requests.get(f"{CHATBOT_API_URL}/health", timeout=5)
        return response.json()
    except:
        return {"status": "unavailable"}

def search_news(query: str, n_results: int = 5) -> list:
    """Search news via API"""
    try:
        response = requests.get(
            f"{CHATBOT_API_URL}/news/search",
            params={"query": query, "n_results": n_results},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("news", [])
        return []
    except:
        return []

def get_recent_news(days: int = 7) -> list:
    """Get recent news via API"""
    try:
        response = requests.get(
            f"{CHATBOT_API_URL}/news/recent",
            params={"days": days, "limit": 20},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("news", [])
        return []
    except:
        return []

# ============================================
# UI Layout
# ============================================
st.title("📈 Stock Prediction Chatbot")

# Sidebar
with st.sidebar:
    st.header("System Status")

    # Health check
    health = get_health_status()
    if health.get("status") == "healthy":
        st.success("API: Online")
        col1, col2 = st.columns(2)
        with col1:
            triton = "✅" if health.get("triton_available") else "❌"
            st.metric("Triton", triton)
        with col2:
            llm = "✅" if health.get("llm_available") else "❌"
            st.metric("LLM", llm)
        st.metric("News Docs", health.get("chromadb_documents", 0))
    else:
        st.error("API: Offline")

    st.divider()

    # Settings
    st.header("Settings")
    api_url = st.text_input("API URL", value=CHATBOT_API_URL)
    if api_url != CHATBOT_API_URL:
        CHATBOT_API_URL = api_url

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.rerun()

    st.divider()

    # Quick Actions
    st.header("Quick Actions")
    if st.button("📰 Show Recent News"):
        st.session_state.show_news = True

# Main content - Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📰 News Search", "📊 Prediction"])

# ============================================
# Tab 1: Chat
# ============================================
with tab1:
    # Chat container
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 Sources"):
                        for src in msg["sources"]:
                            st.markdown(f"- [{src.get('date', 'N/A')}] {src.get('headline', '')} - *{src.get('source', '')}*")

    # Chat input
    if prompt := st.chat_input("Ask about stock market..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_api(prompt)

            st.markdown(response["response"])

            if response.get("sources"):
                with st.expander("📚 Sources"):
                    for src in response["sources"]:
                        st.markdown(f"- [{src.get('date', 'N/A')}] {src.get('headline', '')} - *{src.get('source', '')}*")

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["response"],
            "sources": response.get("sources", [])
        })

    # Example questions
    st.divider()
    st.subheader("💡 Example Questions")
    example_questions = [
        "What's the latest news about the stock market?",
        "How is the tech sector performing?",
        "What are analysts saying about interest rates?",
        "Summarize recent news about AAPL",
        "What factors might affect the S&P 500 this week?"
    ]

    cols = st.columns(3)
    for i, q in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(q, key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

# ============================================
# Tab 2: News Search
# ============================================
with tab2:
    st.subheader("🔍 Search Financial News")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search Query", placeholder="e.g., Federal Reserve interest rate")
    with col2:
        n_results = st.number_input("Results", min_value=1, max_value=50, value=10)

    if st.button("Search", type="primary"):
        if search_query:
            with st.spinner("Searching..."):
                news_results = search_news(search_query, n_results)

            if news_results:
                st.success(f"Found {len(news_results)} results")
                for news in news_results:
                    with st.container():
                        st.markdown(f"""
                        **{news.get('headline', 'No headline')}**

                        📅 {news.get('date', 'N/A')} | 📰 {news.get('source', 'N/A')} | 🏷️ {news.get('symbol', 'N/A')} | Relevance: {news.get('relevance', 0):.2%}
                        """)
                        st.divider()
            else:
                st.warning("No results found")

    st.divider()
    st.subheader("📰 Recent News")

    days = st.slider("Days", 1, 30, 7)
    if st.button("Load Recent News"):
        with st.spinner("Loading..."):
            recent = get_recent_news(days)

        if recent:
            for news in recent:
                st.markdown(f"- [{news.get('date', 'N/A')}] {news.get('headline', '')} - *{news.get('source', '')}*")
        else:
            st.info("No recent news available")

# ============================================
# Tab 3: Prediction
# ============================================
with tab3:
    st.subheader("📊 S&P 500 Price Prediction")

    st.info("""
    This feature requires the Triton Inference Server to be running.
    The model predicts the next 5 days of S&P 500 prices based on:
    - Last 30 days of historical prices
    - Recent news sentiment (via FinBERT embeddings)
    """)

    # Input prices
    st.write("Enter the last 30 days of S&P 500 closing prices:")

    # Option to input manually or use example data
    use_example = st.checkbox("Use example data")

    if use_example:
        # Example: Recent S&P 500 prices (placeholder)
        example_prices = [
            4800, 4815, 4810, 4825, 4840, 4835, 4850, 4865, 4870, 4855,
            4860, 4875, 4880, 4890, 4885, 4900, 4910, 4905, 4920, 4935,
            4940, 4950, 4945, 4960, 4970, 4965, 4980, 4990, 4985, 5000
        ]
        prices_input = st.text_area(
            "Prices (comma-separated)",
            value=", ".join(map(str, example_prices)),
            height=100
        )
    else:
        prices_input = st.text_area(
            "Prices (comma-separated)",
            placeholder="4800, 4815, 4810, ...",
            height=100
        )

    news_query = st.text_input(
        "News Query (optional)",
        placeholder="e.g., S&P 500 outlook"
    )

    if st.button("Predict", type="primary"):
        try:
            prices = [float(p.strip()) for p in prices_input.split(",")]

            if len(prices) != 30:
                st.error(f"Please enter exactly 30 prices. You entered {len(prices)}.")
            else:
                with st.spinner("Making prediction..."):
                    response = requests.post(
                        f"{CHATBOT_API_URL}/predict",
                        json={
                            "prices": prices,
                            "news_query": news_query if news_query else None
                        },
                        timeout=600
                    )

                if response.status_code == 200:
                    result = response.json()
                    predictions = result.get("predictions", [])

                    st.success("Prediction complete!")

                    # Display predictions
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                    with col2:
                        if predictions:
                            change = ((predictions[-1] - prices[-1]) / prices[-1]) * 100
                            st.metric("5-Day Change", f"{change:+.2f}%")

                    st.write("**Predicted Prices (Next 5 Days):**")
                    for i, pred in enumerate(predictions, 1):
                        st.write(f"Day {i}: ${pred:,.2f}")

                    # Show related news if available
                    if result.get("related_news"):
                        st.write("**Related News:**")
                        for news in result["related_news"]:
                            st.markdown(f"- {news.get('headline', '')}")

                elif response.status_code == 503:
                    st.error("Triton server is not available. Please ensure the inference server is running.")
                else:
                    st.error(f"Prediction failed: {response.text}")

        except ValueError:
            st.error("Invalid price format. Please enter comma-separated numbers.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the chatbot API.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================
# Footer
# ============================================
st.divider()
st.caption("⚠️ This is for informational purposes only. Not financial advice.")
