from langchain_community.vectorstores import FAISS

def query_search(vectorstore,query):
  results = vectorstore.similarity_search(query, k=3)
  for r in results:
      print(r.metadata, "\n", r.page_content[:200], "\n---")

  return results