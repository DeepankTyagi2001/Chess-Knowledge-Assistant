# from langchain_community.document_loaders import PyPDFDirectoryLoader

# loader = PyPDFDirectoryLoader(r"C:\Users\deepa\OneDrive\Desktop\Agentic Mastery\RAG\SourceDocs\PDFs")
# docs = loader.load()   

# print(len(docs))
# print(docs[0].page_content)

# print(repr(docs[0].page_content))
# print(len(docs[0].page_content))


# import fitz  # PyMuPDF

# doc = fitz.open(r"C:\Users\deepa\OneDrive\Desktop\Agentic Mastery\RAG\SourceDocs\PDFs\Artur Yusupov - [1] Ian Adams - Build up your Chess 1_ The Fundamentals-Quality Chess (2008).pdf")
# for page in doc:
#     text = page.get_text()
#     print(repr(text[:500]))


# import pdfplumber

# with pdfplumber.open(r"C:\Users\deepa\OneDrive\Desktop\Agentic Mastery\RAG\SourceDocs\PDFs\Artur Yusupov - [1] Ian Adams - Build up your Chess 1_ The Fundamentals-Quality Chess (2008).pdf") as pdf:
#     text = pdf.pages[1].extract_text()
#     print(repr(text))


# ---------------------------------------------------------------------------------------------------------------------------------

import pymupdf4llm
import os
import re
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

pdf_dir = os.getenv("PDF_DOCS_DIRECTORY")

def clean_text(text):

  #remove symbols from starting and ending of images
  text = re.sub(r'<!-- Start of picture text -->.*?<!-- End of picture text -->', '', text, flags=re.DOTALL)

  # Collapse multiple blank lines into one
  text = re.sub(r'\n{3,}', '\n\n', text)
  
  # Collapse multiple spaces
  text = re.sub(r'[ \t]{2,}', ' ', text)
  
  # Strip leading/trailing whitespace per line
  text = '\n'.join(line.strip() for line in text.split('\n'))
  
  return text.strip()


def is_meaningful(text, min_length=30):
    return len(text.strip()) >= min_length



def load_data():
    pdf_dir = os.getenv("PDF_DOCS_DIRECTORY")

    all_docs = []

    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, filename)
        print(f"Processing {filename}...")

        # One call for the WHOLE file
        chunks = pymupdf4llm.to_markdown(  # chunking books into pages
            pdf_path,
            page_chunks=True,
            table_strategy=None,   # skip table detection, not needed for this book
        )

        for chunk in chunks:
            all_docs.append({
                "text": clean_text(chunk["text"]),
                "source": filename,
                "page": chunk["metadata"].get("page"),
            })
            
    return all_docs