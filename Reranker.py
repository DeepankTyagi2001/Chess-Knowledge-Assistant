from sentence_transformers import CrossEncoder

_model = None

def get_reranker():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model

def rerank(query, docs, top_k=3):
    model = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored[:top_k]]