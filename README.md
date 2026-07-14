# RAG Knowledge Base — Sample Document Set

A collection of 15 plain-text `.txt` documents covering five technical topics, purpose-built as sample source material for building and testing a Retrieval-Augmented Generation (RAG) pipeline.

## Purpose

This dataset is designed for experimenting with RAG components: chunking strategies, embedding generation, vector store indexing, semantic search, and answer grounding. Each document is short, self-contained, and topically focused, making it easy to verify whether a retrieval system surfaces the correct source for a given query.

## Dataset Overview

| Metric | Value |
|---|---|
| Total documents | 15 |
| Total topics | 5 |
| Format | Plain text (`.txt`) |
| Word count per file | 377–480 words |
| Total word count | ~6,610 words |
| Total size | ~44 KB |

## Topics & Files

### Python Tutorials
| File | Description |
|---|---|
| `python_basics.txt` | Variables, data types, operators, control flow, and loops |
| `python_functions.txt` | Function definitions, arguments, lambdas, scope, and recursion |
| `python_file_handling.txt` | Reading/writing files, file modes, `with` statements, and exceptions |

### AI Notes
| File | Description |
|---|---|
| `ai_intro.txt` | Overview of AI, subfields, history, and ethical considerations |
| `ai_neural_networks.txt` | Neural network structure, activation functions, and training |
| `ai_nlp.txt` | NLP preprocessing, embeddings, Transformers, and common tasks |

### RAG Notes
| File | Description |
|---|---|
| `rag_intro.txt` | What RAG is, why it's used, and how it reduces hallucinations |
| `rag_embeddings.txt` | Embeddings, chunking, and vector database fundamentals |
| `rag_pipeline.txt` | End-to-end RAG pipeline stages, from ingestion to evaluation |

### Machine Learning Notes
| File | Description |
|---|---|
| `ml_intro.txt` | ML workflow, feature engineering, over/underfitting |
| `ml_supervised_unsupervised.txt` | Supervised vs. unsupervised learning and common algorithms |
| `ml_evaluation_metrics.txt` | Accuracy, precision, recall, F1, RMSE, R², and cross-validation |

### Operating System Notes
| File | Description |
|---|---|
| `os_intro.txt` | Core OS functions, types of operating systems, kernel basics |
| `os_process_management.txt` | Process states, scheduling algorithms, threads, and deadlocks |
| `os_memory_management.txt` | Paging, virtual memory, segmentation, and memory protection |

## Document Design Principles

Each document was written with RAG use cases in mind:

- **Self-contained** — every file can be understood without reading the others, making it a good candidate for a single chunk or a small number of chunks.
- **Bounded length** — capped under 500 words to fit comfortably within typical embedding model input limits and keep chunk counts predictable.
- **Single topic focus** — each file covers one coherent sub-topic, reducing semantic overlap and improving retrieval precision.
- **Plain text, minimal formatting** — no headers, tables, or markup inside the `.txt` files, so text extraction and chunking logic don't need special-case handling.
- **Consistent structure** — each document opens with a short definition/overview, then expands into details, lists, and examples.

## Suggested Usage

1. **Ingest** — load all `.txt` files from this folder into your document loader.
2. **Chunk** — split each document into chunks (e.g., 200–500 tokens, with slight overlap). Given the short length of each file, some pipelines may treat one file as one chunk.
3. **Embed** — generate vector embeddings for each chunk using an embedding model (e.g., OpenAI, Sentence-Transformers, Cohere).
4. **Index** — store the embeddings and original text in a vector database (e.g., FAISS, Chroma, Pinecone, Weaviate).
5. **Query** — test retrieval with sample questions, such as:
   - "What is overfitting in machine learning?"
   - "How does paging work in an operating system?"
   - "What is the difference between precision and recall?"
   - "How do vector databases perform similarity search?"
6. **Evaluate** — check whether the retrieved chunks match the expected source file, and whether the generated answer is grounded in that content.

## File Naming Convention

Files follow a `topic_subtopic.txt` naming pattern (e.g., `ml_evaluation_metrics.txt`), which can also be used as a simple category label or metadata field during ingestion.

## License / Notes

This content was generated as sample/synthetic educational material for testing purposes. It is not sourced from copyrighted material and is free to use, modify, and extend for development and demo purposes.
