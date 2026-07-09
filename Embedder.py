from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def embedder(langchain_docs):
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # cheaper, good quality; use "text-embedding-3-large" for higher accuracy

  vectorstore = FAISS.from_documents(langchain_docs, embeddings)

  return vectorstore
