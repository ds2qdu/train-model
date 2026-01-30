# ============================================
# ChromaDB 간단한 Streamlit UI
# 실행: streamlit run chromadb_ui.py
# ============================================

import streamlit as st
import chromadb
import pandas as pd

# 설정
CHROMADB_PATH = "y:/mlteam-stock-pipeline-storage-pvc-fab68a2b-8953-4b90-a164-6fdf446c9836/chromadb"

st.set_page_config(page_title="ChromaDB Viewer", layout="wide")
st.title("📊 ChromaDB 뉴스 벡터 데이터베이스")

# ChromaDB 연결
@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path=CHROMADB_PATH)

client = get_client()
collection = client.get_collection("stock_news")

# 사이드바
st.sidebar.header("정보")
total_count = collection.count()
st.sidebar.metric("총 문서 수", f"{total_count:,}개")

# 탭
tab1, tab2, tab3 = st.tabs(["📋 전체 조회", "🔍 검색", "📈 통계"])

with tab1:
    st.subheader("최근 뉴스")
    limit = st.slider("표시 개수", 10, 100, 20)

    data = collection.peek(limit=limit)

    if data['ids']:
        df = pd.DataFrame({
            'ID': data['ids'],
            'Headline': data['documents'],
            'Date': [m.get('date', '') for m in data['metadatas']],
            'Source': [m.get('source', '') for m in data['metadatas']],
            'Symbol': [m.get('symbol', '') for m in data['metadatas']]
        })
        st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("뉴스 검색")

    col1, col2 = st.columns(2)

    with col1:
        search_type = st.radio("검색 방식", ["텍스트 유사도", "날짜별", "소스별"])

    with col2:
        if search_type == "텍스트 유사도":
            query = st.text_input("검색어", "stock market")
            n_results = st.slider("결과 수", 5, 50, 10)

            if st.button("검색"):
                results = collection.query(query_texts=[query], n_results=n_results)
                if results['ids'][0]:
                    df = pd.DataFrame({
                        'Headline': results['documents'][0],
                        'Date': [m.get('date', '') for m in results['metadatas'][0]],
                        'Source': [m.get('source', '') for m in results['metadatas'][0]],
                        'Distance': results['distances'][0]
                    })
                    st.dataframe(df, use_container_width=True)

        elif search_type == "날짜별":
            date = st.text_input("날짜 (YYYY-MM-DD)", "2026-01-26")
            if st.button("조회"):
                results = collection.get(where={"date": date}, include=["documents", "metadatas"])
                if results['ids']:
                    df = pd.DataFrame({
                        'Headline': results['documents'],
                        'Source': [m.get('source', '') for m in results['metadatas']]
                    })
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("해당 날짜에 뉴스가 없습니다.")

        elif search_type == "소스별":
            source = st.text_input("소스명", "Yahoo")
            if st.button("조회"):
                results = collection.get(where={"source": source}, include=["documents", "metadatas"])
                if results['ids']:
                    df = pd.DataFrame({
                        'Headline': results['documents'],
                        'Date': [m.get('date', '') for m in results['metadatas']]
                    })
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("해당 소스의 뉴스가 없습니다.")

with tab3:
    st.subheader("통계")

    all_data = collection.get(include=["metadatas"])

    # 날짜별 통계
    st.write("**날짜별 뉴스 개수**")
    date_counts = {}
    for meta in all_data['metadatas']:
        date = meta.get('date', 'unknown')
        date_counts[date] = date_counts.get(date, 0) + 1

    df_dates = pd.DataFrame(
        sorted(date_counts.items(), reverse=True)[:20],
        columns=['Date', 'Count']
    )
    st.bar_chart(df_dates.set_index('Date'))

    # 소스별 통계
    st.write("**소스별 뉴스 개수**")
    source_counts = {}
    for meta in all_data['metadatas']:
        source = meta.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1

    df_sources = pd.DataFrame(
        sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        columns=['Source', 'Count']
    )
    st.bar_chart(df_sources.set_index('Source'))
