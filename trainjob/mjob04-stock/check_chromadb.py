# ============================================
# ChromaDB 벡터 데이터베이스 확인 스크립트
# ============================================

import chromadb
from datetime import datetime

# PVC 마운트 경로
CHROMADB_PATH = "y:/mlteam-stock-pipeline-storage-pvc-fab68a2b-8953-4b90-a164-6fdf446c9836/chromadb"

def main():
    print("=" * 70)
    print("  ChromaDB 벡터 데이터베이스 확인")
    print("=" * 70)

    # 1. ChromaDB 연결
    print(f"\n[1] ChromaDB 연결: {CHROMADB_PATH}")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)

    # 2. 컬렉션 목록
    print("\n[2] 컬렉션 목록:")
    collections = client.list_collections()
    for col in collections:
        print(f"    - {col.name}")

    # 3. stock_news 컬렉션 상세
    print("\n[3] stock_news 컬렉션 상세:")
    collection = client.get_collection("stock_news")

    total_count = collection.count()
    print(f"    총 문서 수: {total_count:,}개")

    # 4. 샘플 데이터 확인
    print("\n[4] 최근 저장된 뉴스 (10개):")
    print("-" * 70)

    sample = collection.peek(limit=10)
    for i in range(len(sample['ids'])):
        doc = sample['documents'][i] if sample['documents'] else ""
        meta = sample['metadatas'][i] if sample['metadatas'] else {}

        print(f"  ID: {sample['ids'][i]}")
        print(f"  날짜: {meta.get('date', 'N/A')}")
        print(f"  소스: {meta.get('source', 'N/A')}")
        print(f"  심볼: {meta.get('symbol', 'N/A')}")
        print(f"  헤드라인: {doc[:100]}{'...' if len(doc) > 100 else ''}")
        print("-" * 70)

    # 5. 날짜별 통계
    print("\n[5] 날짜별 뉴스 개수 (최근 10일):")

    # 모든 메타데이터 가져오기
    all_data = collection.get(include=["metadatas"])
    date_counts = {}
    for meta in all_data['metadatas']:
        date = meta.get('date', 'unknown')
        date_counts[date] = date_counts.get(date, 0) + 1

    # 정렬하여 최근 10일 출력
    sorted_dates = sorted(date_counts.items(), reverse=True)[:10]
    for date, count in sorted_dates:
        print(f"    {date}: {count}개")

    # 6. 소스별 통계
    print("\n[6] 뉴스 소스별 개수:")
    source_counts = {}
    for meta in all_data['metadatas']:
        source = meta.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1

    sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for source, count in sorted_sources:
        print(f"    {source}: {count}개")

    # 7. 특정 날짜 뉴스 조회 예시
    print("\n[7] 특정 날짜 뉴스 조회 방법:")
    print("""
    # 특정 날짜 뉴스 조회
    results = collection.get(where={"date": "2026-01-26"})

    # 특정 심볼 뉴스 조회
    results = collection.get(where={"symbol": "AAPL"})

    # 유사 뉴스 검색 (임베딩 기반)
    results = collection.query(
        query_texts=["stock market crash"],
        n_results=5
    )
    """)

    # 8. 임베딩 차원 확인
    print("[8] 임베딩 정보:")
    sample_with_emb = collection.peek(limit=1)
    if sample_with_emb['embeddings']:
        emb_dim = len(sample_with_emb['embeddings'][0])
        print(f"    임베딩 차원: {emb_dim}")
    else:
        print("    임베딩 정보 없음")

    print("\n" + "=" * 70)
    print("  확인 완료")
    print("=" * 70)


def search_news(query_text, n_results=5):
    """텍스트로 유사 뉴스 검색"""
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_collection("stock_news")

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    print(f"\n'{query_text}' 관련 뉴스:")
    print("-" * 70)
    for i in range(len(results['ids'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i] if results['distances'] else 0

        print(f"  [{meta.get('date', 'N/A')}] {doc[:80]}...")
        print(f"  거리: {distance:.4f} | 소스: {meta.get('source', 'N/A')}")
        print()


def get_news_by_date(date_str):
    """특정 날짜 뉴스 조회"""
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_collection("stock_news")

    results = collection.get(
        where={"date": date_str},
        include=["documents", "metadatas"]
    )

    print(f"\n{date_str} 뉴스 ({len(results['ids'])}개):")
    print("-" * 70)
    for i, doc in enumerate(results['documents'][:10]):
        meta = results['metadatas'][i]
        print(f"  [{meta.get('source', 'N/A')}] {doc[:80]}...")


if __name__ == "__main__":
    main()

    # 추가 예시 (주석 해제하여 사용)
    # search_news("Federal Reserve interest rate")
    # get_news_by_date("2026-01-26")
