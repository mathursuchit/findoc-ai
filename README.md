# FinDoc AI — Financial Document Analyst

A RAG-based chatbot for querying financial documents. Upload SEC 10-K filings, earnings reports, or annual reports and ask questions in plain English.

**Live demo:** [mathursuchit-findoc-ai.streamlit.app](https://mathursuchit-findoc-ai.streamlit.app)

---

## Why I built this

I work in financial services and constantly deal with dense regulatory filings and earnings reports. Reading a 200-page 10-K to find one number is painful. I wanted something that lets me ask questions directly and get answers with page-level attribution so I can verify them.

The bigger motivation was learning how RAG actually works under the hood — not just calling an API, but understanding the retrieval pipeline and where it breaks down.

---

## How it works

```
PDF upload
  → split into 1000-char chunks (200 overlap)
  → embed with all-MiniLM-L6-v2 (local, no API cost)
  → store in ChromaDB (persisted to disk)

User question
  → rewrite using chat history (handles "what about that?" follow-ups)
  → MMR retrieval — top 5 chunks across all uploaded docs
  → Llama 3.3 70B via Groq generates answer from retrieved chunks
  → streams back token by token with page-level source attribution
```

The trickiest part was cross-document retrieval. Plain similarity search kept returning all chunks from the same document. Switching to MMR (Maximum Marginal Relevance) fixed this by penalizing redundant chunks and forcing diversity.

---

## Stack

- **LangChain** — RAG pipeline, prompt templates, LCEL chains
- **Groq + Llama 3.3 70B** — LLM inference (free tier, very fast)
- **ChromaDB** — vector store, persisted locally
- **HuggingFace all-MiniLM-L6-v2** — sentence embeddings (runs locally)
- **Streamlit** — UI and deployment

---

## Features

- Multi-document support — upload multiple filings and query across all of them
- Persistent vector store — re-open the app without re-indexing
- Streaming responses
- Source attribution — every answer shows the source file and page number
- Chat history — handles follow-up questions in context

---

## Run locally

```bash
git clone https://github.com/mathursuchit/findoc-ai
cd findoc-ai
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

```bash
streamlit run app.py
```

---

## What I'd improve

- Add RAGAS evaluation to measure faithfulness and retrieval precision automatically
- Hybrid search (BM25 + vector) for better recall on exact terms like ticker symbols
- Table extraction — PyPDF misses financial tables, would need a specialized parser
- TODO: add a document comparison mode (e.g. compare two years of the same 10-K)
