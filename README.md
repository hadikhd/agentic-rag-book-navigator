# Manly P. Hall RAG

> A book-grounded Retrieval-Augmented Generation system for the esoteric and
> philosophical works of **Manly P. Hall**. Accepts a Persian (or mixed) question,
> retrieves grounded evidence from indexed books, generates a structured scholarly
> answer with local LLMs, and returns it in fluent Persian — with built-in RAGAS
> quality evaluation.

---

## ✨ Features

- **Persian-first Q&A** — ask in Persian; the system translates, retrieves in English, and answers in polished scholarly Persian.
- **Book-grounded** — answers are strictly grounded in the indexed books (no fabrication).
- **Structured 4-section answer** — English Answer (Persian body), Sources in the Book, Search Guidance, Related Topics.
- **Multi-agent pipeline** orchestrated with LangGraph over a shared `RAGState`.
- **Local-first** — runs entirely on local Ollama models + local embeddings; no paid APIs.
- **Book & topic navigation** — sidebar dashboard, topic tree, mind map, and study plan per book.
- **RAGAS evaluation** — faithfulness, answer relevancy, context precision, and more, with a metrics dashboard.
- **Resilient** — every stage has fallback behavior (heuristic routing, unfiltered retrieval, per-section writer recovery).

---

## 🧱 Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph |
| LLMs & Embeddings | Ollama (qwen2.5, deepseek-r1, bge-m3, …) |
| Vector Store | Qdrant |
| LLM Framework | LangChain |
| UI | Streamlit |
| Evaluation | RAGAS |
| PDF Processing | PyMuPDF |

---

## 🏗️ Architecture

The system is organized into **three blocks**:

```
1. INGESTION (offline)   PDF → clean → topics index + hierarchy → chunk → enrich → embed → Qdrant
2. ASKING (online)       query → translator → enhancer → router → retriever → reranker
                              → compressor → writer → persian_translator → answer
3. EVALUATION (post)     (question, answer, contexts) → RAGAS → scores → JSON storage → dashboard
```

**Pipeline (LangGraph):**

```
translator → enhancer → router → retriever → reranker → compressor → writer → persian_translator → END
```

Each node reads/writes a shared `RAGState`. Evaluation runs as a **UI-side hook**
after the pipeline returns — it is not a graph node.

> 📖 For a full file-by-file reference and detailed data-flow diagrams, see
> [`Project-Map.md`](./Project-Map.md).

---

## 📁 Project Structure

```
RAG/
├── app/
│   ├── agents/            # Pipeline agents (translator, enhancer, router, retriever,
│   │                      #   reranker, compressor, writer, persian_translator, evaluation)
│   ├── config/            # Settings + esoteric EN→FA glossary
│   ├── graph/             # LangGraph: state, nodes, graph wiring
│   ├── ingestion/         # PDF loading, chunking, TOC, hierarchy, embedding, Qdrant writer
│   ├── llm/               # Ollama client wrapper
│   ├── services/          # run_rag_pipeline() + glossary service
│   ├── vectorstore/       # Qdrant helpers
│   └── utils/
├── evaluation/            # RAGAS evaluator + JSON storage + results
├── scripts/               # ingest_books.py CLI runner
├── data/                  # raw_books/ + processed_books/ + vectorstore/
├── fonts/                 # Inter + Vazirmatn fonts for the UI
├── streamlit_app.py       # The Streamlit UI (entry point)
├── docker-compose.yaml    # Qdrant container
├── requirements.txt
└── Project-Map.md         # Detailed file-by-file documentation
```

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (`http://localhost:11434`)
- [Docker](https://www.docker.com/) (for Qdrant)

### 1. Clone & install

```bash
git clone <your-repo-url>
cd RAG

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull required Ollama models

```bash
ollama pull qwen2.5:7b-instruct
ollama pull deepseek-r1
ollama pull bge-m3:latest
ollama pull translategemma:4b
```

### 3. Start Qdrant

```bash
docker compose up -d
```

### 4. Ingest books

Place PDFs in `data/raw_books/`, then:

```bash
python scripts/ingest_books.py
```

This builds the topics index, chunks, embeds, and stores everything in Qdrant.

### 5. Launch the UI

```bash
streamlit run streamlit_app.py
```

Open the printed URL (typically `http://localhost:8501`).

---

## ⚙️ Configuration

All settings are environment-driven and defined in [`app/config/settings.py`](./app/config/settings.py).

### Key settings

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint. |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant connection. |
| `QDRANT_COLLECTION` | `manly_hall_books` | Vector collection name. |
| `TRANSLATOR_MODEL` | `translategemma:4b` | Query translation model. |
| `ENHANCER_MODEL` | `qwen2.5:7b-instruct` | Query enhancement model. |
| `ROUTER_MODEL` | `qwen2.5:7b-instruct` | Topic routing model. |
| `WRITER_MODEL` | `deepseek-r1` | Answer generation model. |
| `PERSIAN_TRANSLATOR_MODEL` | `translategemma:12b` | Final Persian translation model. |
| `EMBEDDING_MODEL` | `bge-m3:latest` | Embedding model. |
| `RETRIEVER_TOP_K` | `10` | Number of chunks to retrieve. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Chunking parameters. |
| `USE_MULTI_BOOK` | `True` | Multi-book vs single-book mode. |
| `EVAL_ENABLED` | `True` | Enable RAGAS evaluation. |
| `EVAL_METRICS` | `["faithfulness"]` | Active RAGAS metrics. |

Create a `.env` file in the project root to override any of these.

---

## 📊 Evaluation

After each answer, the UI runs RAGAS evaluation and stores results under
`evaluation/results/evaluations_YYYYMMDD.json`.

Available metrics:

| Metric | Needs `ground_truth`? |
|--------|-----------------------|
| `faithfulness` | No |
| `answer_relevancy` | No |
| `context_precision` | No |
| `context_recall` | **Yes** |
| `answer_similarity` | **Yes** |
| `answer_correctness` | **Yes** |

> **Note:** `ground_truth` is currently empty, so the last three metrics are not
> meaningful until reference answers are provided.

The **Evaluation Dashboard** tab in the UI shows metric trends, a system quality
score, per-record drilldowns, and threshold alerts.

---

## 🧪 Usage

### Ask a question (UI)

1. Open the Streamlit app.
2. Pick a book and (optionally) a topic from the sidebar.
3. Type your question (Persian or English) in the **Ask Book** tab.
4. View the structured answer: main answer, sources, search guidance, related topics.
5. Toggle **Debug mode** to inspect the full pipeline state.

### Programmatic access

```python
from app.services.rag_pipeline import run_rag_pipeline

result = run_rag_pipeline(
    user_query="نماد اسکاراب در سنت مصری چه معنایی دارد؟",
    book_id="manly-p-hall-secret-teachings-of-all-ages",
)

print(result["translated_answer"])  # final Persian answer
```

---

## 📚 Documentation

- [`Project-Map.md`](./Project-Map.md) — detailed file-by-file reference, data-flow
  diagrams, per-agent I/O tables, and the three-block architecture.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `connection refused` on Ollama | Ensure `ollama serve` is running. |
| Qdrant connection error | Run `docker compose up -d`. |
| Empty retrieval results | Re-run ingestion; check `QDRANT_COLLECTION` matches. |
| Slow evaluation | Reduce `EVAL_METRICS` or set `EVAL_SAMPLING_RATE < 1.0`. |
| Persian text not rendering | Ensure `fonts/` directory is present. |

---

## 📄 License

[Add your license here — e.g., MIT, Apache-2.0, or "All rights reserved".]

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like
to change.
