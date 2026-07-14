from langchain_community.retrievers import BM25Retriever

def build_bm25_retriever(langchain_docs, k=5):
    retriever = BM25Retriever.from_documents(langchain_docs)
    retriever.k = k
    return retriever