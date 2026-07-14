from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

async def embedder(langchain_docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = await FAISS.afrom_documents(langchain_docs, embeddings)  # afrom_documents, not from_documents
    return vectorstore