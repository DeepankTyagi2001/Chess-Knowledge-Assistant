# Chess Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system that answers chess questions grounded in a library of 10 chess books, using hybrid retrieval (dense + sparse), multi-query expansion, and cross-encoder reranking.

## Pipeline Overview

```
                                   ┌─────────────────────┐
                                   │   PDF chess books    │
                                   └──────────┬───────────┘
                                              │
                                   LoadingPDFs.py (pymupdf4llm,
                                   page-wise extraction + regex clean)
                                              │
                                              ▼
                                   Chunker.py (RecursiveCharacterTextSplitter,
                                   800 chars / 100 overlap)
                                              │
                              ┌───────────────┴────────────────┐
                              ▼                                 ▼
                    Embedder.py (async,                HybridRetriever.py
                    OpenAI text-embedding-3-small,      (BM25, sparse/keyword)
                    FAISS.afrom_documents)
                              │                                 │
                              ▼                                 │
                    faiss_index_chess/ (saved)         built fresh each run
                    chunked_docs.pkl (saved)            from chunked_docs.pkl
                              │                                 │
                              └───────────────┬─────────────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │   User query        │
                                    └─────────┬───────────┘
                                              │
                                   QueryExpander.py (gpt-4o-mini,
                                   generates 3 reformulated variants
                                   + keeps original = 4 queries total)
                                              │
                                              ▼
                                   Query_Search.py → hybrid_multi_query_search()
                                   (dense FAISS + sparse BM25 per variant,
                                   merged + deduped by source/page/chunk_index)
                                              │
                                              ▼
                                   Reranker.py (cross-encoder
                                   ms-marco-MiniLM-L-6-v2, reranks
                                   against the ORIGINAL query, top 3)
                                              │
                                              ▼
                                   call_llm.py (gpt-4o-mini,
                                   grounded system prompt, final answer)
```

## File Structure

| File                 | Responsibility                                                                                                                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LoadingPDFs.py`     | Extracts text page-wise from all PDFs in `PDF_DOCS_DIRECTORY` using `pymupdf4llm`, cleans it with regex (strips image placeholders, collapses whitespace)                                           |
| `Chunker.py`         | Splits cleaned page text into ~800-character chunks (100 overlap) via `RecursiveCharacterTextSplitter`, tags each with `source`, `page`, `chunk_index` metadata                                     |
| `Embedder.py`        | Async-embeds chunks with OpenAI `text-embedding-3-small` and builds a FAISS index (`FAISS.afrom_documents`)                                                                                         |
| `HybridRetriever.py` | Builds a BM25 sparse retriever from the same chunks, for keyword-exact matches (opening names, notation) that embeddings can miss                                                                   |
| `QueryExpander.py`   | Given a user question, asks `gpt-4o-mini` to generate 3 reformulated variants; returns original + variants (4 queries total)                                                                        |
| `Query_Search.py`    | `query_search()` — single-query dense search (kept for baseline comparison). `hybrid_multi_query_search()` — runs dense + sparse retrieval per query variant, merges and dedupes results            |
| `Reranker.py`        | Loads a local cross-encoder (`ms-marco-MiniLM-L-6-v2`) and reranks the merged candidate pool against the **original** query, returning the final top-k                                              |
| `call_llm.py`        | Builds the grounded system prompt (chess-specific rules: notation formatting, no outside knowledge, surface contradictions, no hallucinated moves) and generates the final answer via `gpt-4o-mini` |
| `LoadQuestion.py`    | Parses a numbered `.txt` list of questions into a plain list of strings                                                                                                                             |
| `queries_list.txt`   | 25 manually written test questions spanning openings, tactics, rules, and book-specific content                                                                                                     |
| `main.py`            | Orchestrates the full pipeline: builds or loads the index, builds BM25, loops through test questions, retrieves, and generates answers                                                              |
| `FutureTasks.txt`    | Running roadmap of planned improvements                                                                                                                                                             |

## Setup

### Dependencies

```bash
pip install langchain langchain-openai langchain-community faiss-cpu \
            pymupdf4llm sentence-transformers rank_bm25 openai python-dotenv
```

### Environment variables (`.env`)

```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=your_base_url_here      # optional, only if using a proxy/alt endpoint
PDF_DOCS_DIRECTORY=path/to/your/pdf/folder
```

### Running

```bash
python main.py
```

On first run (no `faiss_index_chess/` or `chunked_docs.pkl` present), the pipeline extracts and chunks all PDFs, embeds them, and saves both the FAISS index and the raw chunks to disk. Subsequent runs load from disk and skip straight to retrieval — the chunks pickle exists specifically so BM25 can be rebuilt without re-extracting or re-embedding.

## Design Decisions Worth Knowing

- **Extraction is intentionally synchronous.** An earlier version parallelized PDF extraction across CPU processes, but with image-heavy chess books this saturated memory and CPU without finishing. Since extraction only runs once (cached to disk afterward), the time cost doesn't matter enough to justify the complexity — synchronous is correct here.
- **Embedding is async** (`Embedder.py` uses `FAISS.afrom_documents`, which batches calls via `aembed_documents`) since this is a genuinely I/O-bound step that benefits from concurrency, and it's the step most likely to get re-triggered during iteration.
- **Reranking always scores against the original query, not the expanded variants.** Variants exist to widen the retrieval net; final relevance judgment should reflect what the user actually asked.
- **Dense and sparse results are merged as an equal-weight union**, not fused with a weighted scheme like RRF. The reranker is expected to do the real relevance sorting on the combined pool — weighted fusion is a possible future addition if reranking turns out insufficient to filter out weak BM25 matches.

## Known Gaps / Not Yet Implemented

- **No evaluation set.** There's currently no ground-truth mapping from questions to expected chunks/answers, so retrieval and generation quality changes can't be measured objectively yet — only eyeballed. This is the highest-priority next step before further tuning retrieval.
- **No result logging.** `main.py` prints answers to console but doesn't persist them; results aren't saved for later comparison across pipeline versions.
- **LLM-based text cleaning** (replacing the regex cleaner in `LoadingPDFs.py`) is deferred — regex handles the current book formatting adequately for now.
- **Move-sequence search** ("search based on a set of moves played") is an open idea, not yet designed or implemented.

## Roadmap

See `FutureTasks.txt` for the full running list. Completed so far: async embedding, reranker, query expander, multi-query generation, hybrid (dense + sparse) retrieval, grounded system prompt. Remaining: eval set, result logging, optional LLM-based cleaning, move-sequence search.
