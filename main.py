import os
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from LoadingPDFs import load_data
from Chunker import chunker
from Embedder import embedder
from Query_Search import query_search
from call_llm import call_llm
from LoadQuestion import load_questions

FAISS_INDEX_PATH = "faiss_index_chess"


async def main():
    questions = load_questions("queries_list.txt")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index from disk...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("No existing index found. Building new FAISS index...")
        all_docs = load_data()              # now async
        langchain_docs = chunker(all_docs)
        vectorstore = await embedder(langchain_docs)  # now async
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"Saved new index to {FAISS_INDEX_PATH}")

    for i, query in enumerate(questions, start=1):
        print(f"Running question {i}: {query}")
        print("--" * 75)
        context = query_search(vectorstore, query)
        print(call_llm(query, context))
        print("**" * 75)


if __name__ == "__main__":
    asyncio.run(main())