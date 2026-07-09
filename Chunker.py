from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 

from LoadingPDFs import is_meaningful

def chunker (all_docs):
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=800,        # tune based on your content; 500-1000 chars is typical
      chunk_overlap=100,     # preserves context across chunk boundaries
      separators=["\n\n", "\n", ". ", " ", ""],
  )

  langchain_docs = []
  for doc in all_docs:
      if not is_meaningful(doc["text"]):
          continue
      chunks = splitter.split_text(doc["text"])
      for i, chunk in enumerate(chunks):
          langchain_docs.append(
              Document(
                  page_content=chunk,
                  metadata={
                      "source": doc["source"],
                      "page": doc["page"],
                      "chunk_index": i,
                  }
              )
          )

  print(f"Total chunks: {len(langchain_docs)}")
  
  return langchain_docs