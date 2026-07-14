from langchain_community.vectorstores import FAISS

def query_search(vectorstore, query, k=3):
    """Original single-query dense search — kept for baseline comparison."""
    return vectorstore.similarity_search(query, k=k)

def hybrid_multi_query_search(vectorstore, bm25_retriever, queries, k_per_query=5):
    """For each query variant, pulls dense (FAISS) + sparse (BM25) candidates, merges and dedupes."""
    seen = set()
    merged = []

    for q in queries:
        dense_results = vectorstore.similarity_search(q, k=k_per_query)
        sparse_results = bm25_retriever.invoke(q)

        for doc in dense_results + sparse_results:
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.metadata.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                merged.append(doc)

    return merged