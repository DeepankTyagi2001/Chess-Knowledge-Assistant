from langchain_community.vectorstores import FAISS

def query_search(vectorstore, query, k=3):
    """Original single-query search — kept for baseline comparison."""
    return vectorstore.similarity_search(query, k=k)

def multi_query_search(vectorstore, queries, k_per_query=5):
    """Retrieves for each query variant, merges and dedupes by source+page+chunk_index."""
    seen = set()
    merged = []
    for q in queries:
        results = vectorstore.similarity_search(q, k=k_per_query)
        for doc in results:
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.metadata.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged