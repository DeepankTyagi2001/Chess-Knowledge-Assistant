import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from LoadingPDFs import load_data
from Chunker import chunker
from Embedder import embedder
from Query_Search import query_search
from call_llm import call_llm

FAISS_INDEX_PATH = "faiss_index_chess"

def main():
    
    query= input("Enter Your Query:\n")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index from disk...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("No existing index found. Building new FAISS index...")

        all_docs = load_data()            # got page wise chunked data

        langchain_docs = chunker(all_docs)  # got actual chunked data

        vectorstore = embedder(langchain_docs)

        # Save locally so you don't have to re-embed every time
        vectorstore.save_local("faiss_index_chess")

        print(f"Saved new index to {FAISS_INDEX_PATH}")

    context = query_search(vectorstore, query)

    print(call_llm(query, context))


if __name__=="__main__":
  main()