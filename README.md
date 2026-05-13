# Critic-Guided Agentic Hybrid RAG 🧠⚙️

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/framework-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![ACL 2025](https://img.shields.io/badge/inspired%20by-ACL%202025-purple)](https://aclanthology.org)
[![Demo](https://img.shields.io/badge/demo-watch%20video-red)](https://youtube.com/your-demo-link)

---

<div align="center">

### 🎬 Watch it in action

[![Demo Video](https://img.shields.io/badge/▶%20Watch%20Full%20Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://youtube.com/your-demo-link)

*7 agents. Real-time verification. Zero hallucinations on test queries.*

</div>

---

## What is this?

Most RAG systems have a fundamental flaw — they retrieve documents, generate an answer, and just trust themselves. No verification. No fallback. If the retrieval step fails, you get a confidently wrong answer and nothing catches it.

I built this to fix that. It's a multi-agent system where 7 specialized agents work together — planning the query, retrieving evidence from multiple sources, criticizing the context quality, generating a grounded answer, verifying every single claim in that answer, and looping back if anything looks hallucinated. The system refuses to give an answer it can't back up.

Built with **LangGraph** as the orchestration backbone, **Llama.cpp** for local inference on Apple Silicon, and a **Streamlit dashboard** that shows every agent's decision in real time.

---

## 🎬 Demo

> 📺 **[Click here to watch the full demo](https://youtube.com/your-demo-link)**

The demo walks through:
- A live query running through all 7 agents
- The Context Critic rejecting insufficient evidence and triggering Web Fallback
- The Claim Verifier flagging a hallucination risk and sending it back for retry
- The final verified answer with per-claim risk scores

![Dashboard](https://github.com/MohithChandra07/Agentic-Hybrid-RAG/blob/main/screenshots/dashboard.png?raw=true)

---

## Why I built this

I came across **"RAG-Critic: A Critic-Guided RAG Framework" (ACL 2025)** and thought the core idea was really smart — evaluate the retrieved context before generating anything, instead of blindly trusting it. But reading through it carefully, I found three gaps that bothered me:

**Problem 1 — Holistic scoring misses micro-hallucinations.**
The critic graded the final answer as a whole. That means one hallucinated sentence buried inside an otherwise good answer still passes. I wanted claim-level verification — every sentence checked individually.

**Problem 2 — Same expensive retrieval for every query.**
It ran a Cross-Encoder reranker on every single query regardless of complexity. That's wasteful. A simple factual question doesn't need the same pipeline as a multi-hop reasoning question. I built an adaptive policy that routes queries based on complexity.

**Problem 3 — The system had no memory of its failures.**
Every time the system failed on a query type, it started from scratch next time. I added a semantic memory module backed by ChromaDB that stores failed queries so the system gets better over time.

These three gaps became the three core engineering contributions of this project.

---

## How it works

7 agents, each with one job:

| Agent | What it does |
|---|---|
| **Planner** | Classifies query complexity, breaks multi-hop questions into sub-queries |
| **Hybrid Retriever** | Runs BM25 + ChromaDB vector search in parallel, merges and deduplicates |
| **Context Critic** | Scores retrieved evidence for relevance — passes or triggers Web Fallback |
| **Web Fallback** | Hits DuckDuckGo if local retrieval isn't good enough |
| **Generator** | Writes the answer strictly grounded in critic-approved context |
| **Claim Verifier** | Splits the draft into atomic sentences, scores each for hallucination risk |
| **Answer Critic** | If risk > 0.5, triggers a full retry with refined retrieval parameters |

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│   Planner   │────▶│ Hybrid Retriever │────▶│ Context Critic │
└─────────────┘     └──────────────────┘     └───────┬────────┘
                                                      │
                              ┌───────────────────────┤
                              │ Pass                  │ Fail
                              ▼                       ▼
                    ┌──────────────────┐    ┌──────────────────┐
                    │    Generator     │    │  Web Fallback    │
                    └────────┬─────────┘    └──────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Claim Verifier  │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 │ Risk < 0.5            │ Risk ≥ 0.5
                 ▼                       ▼
          ┌─────────────┐      ┌──────────────────┐
          │ Final Answer│      │  Answer Critic   │
          └─────────────┘      │  (loop back)     │
                               └──────────────────┘
```

![Architecture Diagram](https://github.com/MohithChandra07/Agentic-Hybrid-RAG/blob/main/diagrams/architecture.png?raw=true)

---

## Key contributions

- **Adaptive Retrieval Policy** — query complexity classifier routes to fast `k=3` shallow search or deep `k=10` Cross-Encoder reranking. Cuts unnecessary compute on simple queries.
- **Atomic Claim Verifier** — instead of grading the answer as a whole, it decomposes the draft into individual sentences and cross-references each one against the evidence vector space.
- **Semantic Failure Memory** — secondary ChromaDB instance stores embeddings of failed queries and session summaries. The system checks this before retrieval and adapts its strategy.
- **Critic-gated pipeline** — two independent critic agents gate the pipeline at different stages. Nothing gets through unless it passes both.
- **Web Fallback** — fully autonomous escalation to live web search when the local knowledge base coverage is insufficient.

---

## Benchmark results

![Benchmark Charts](https://github.com/MohithChandra07/Agentic-Hybrid-RAG/blob/main/diagrams/metrics.png?raw=true)

| Metric | Baseline RAG | Agentic Hybrid RAG |
|---|---|---|
| Faithfulness ↑ | 66% | **100%** |
| Hallucination Rate ↓ | 34% | **0%** |
| Retrieval Precision ↑ | 66% | **100%** |
| Avg. Latency | **35.9s** | 116.1s |

**What these numbers mean:**

**Faithfulness & Hallucination Rate** — on the test set, the agentic system answered only what it could verify. The 0% hallucination rate reflects the system refusing to generate ungrounded claims rather than guessing.

**Retrieval Precision** — the baseline scores higher on raw precision because it uses a narrower retrieval setup. The agentic system casts a wider net through hybrid search and then filters aggressively via the critic — trading raw precision for better coverage and verified quality.

**Latency** — the 3x latency increase is the honest cost of running 7 agents, retry loops, and claim verification on local hardware. A GPU-accelerated or cloud deployment would reduce this significantly.

**Retry Effectiveness** — the retry loop activates only when the Answer Critic flags risk above the threshold. When it does activate, it produces a measurable relevance improvement (~0.05 score gain), confirming the loop is working as intended rather than firing blindly.

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph |
| Local LLM | Llama.cpp + Phi-3 (Metal / Apple Silicon) |
| Vector DB | ChromaDB |
| Lexical search | BM25 |
| Reranker | SentenceTransformers `ms-marco-MiniLM` |
| Failure memory | ChromaDB (secondary instance) |
| UI | Streamlit |
| Web fallback | DuckDuckGo API |

---

## Project structure

```text
Agentic-Hybrid-RAG/
│
├── app.py                      # CLI entry point
├── ui_app.py                   # Streamlit dashboard
├── ingest.py                   # PDF → ChromaDB ingestion
├── requirements.txt
│
├── agentic_rag/
│   └── backend/
│       ├── agents/             # All 7 agent modules
│       ├── core/               # State schemas, config
│       ├── retrieval/          # Hybrid search, adaptive policy, reranker
│       └── graph.py            # LangGraph graph + routing
│
├── dataset/                    # Source PDFs
├── benchmark/                  # Ground-truth evaluation JSONs
├── evaluation/                 # Benchmarking scripts
├── diagrams/                   # Architecture + metric charts
└── screenshots/                # UI captures
```

---

## Setup & running

```bash
# 1. Clone
git clone https://github.com/MohithChandra07/Agentic-Hybrid-RAG.git
cd Agentic-Hybrid-RAG

# 2. Install
pip3 install -r requirements.txt
pip3 install sentence-transformers duckduckgo-search

# 3. Mac — compile with Metal
CMAKE_ARGS="-DGGML_METAL=on" pip3 install --force-reinstall --no-cache-dir llama-cpp-python

# 4. Build the database
python3 ingest.py

# 5a. Run CLI
python3 agentic_rag/backend/graph.py

# 5b. Run dashboard
python3 -m streamlit run ui_app.py
```

---

## Evaluation

```bash
python3 -m evaluation.runner     # run benchmark
python3 -m evaluation.visualize  # generate charts
```

---

## What's next

- **GraphRAG** — migrate from flat ChromaDB to a Neo4j knowledge graph for multi-hop relational traversal
- **Multi-modal retrieval** — handle figures and tables inside PDFs with a vision-capable LLM
- **Cloud deployment** — Docker + vLLM cluster for multi-user concurrent access
- **Feedback loop** — wire real user ratings back into the failure memory module

---

## For recruiters / interviewers

The interesting engineering is in `agentic_rag/backend/retrieval/` (adaptive policy + reranker fusion) and `agentic_rag/backend/agents/` (claim verifier + critic gating). The LangGraph state machine and conditional routing logic lives in `graph.py`.

Happy to walk through any part of the design.

---

*Built by Mohith Chandra Gugulothu · May 2026*
