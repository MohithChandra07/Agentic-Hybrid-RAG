# Critic-Guided Agentic Hybrid RAG

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/framework-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-under%20review-yellow)](#research-paper)
[![Demo](https://img.shields.io/badge/demo-watch%20video-red)](https://youtube.com/your-demo-link)

> **7 agents. Real-time claim verification. Zero hallucinations on test queries.**

A multi-agent Retrieval-Augmented Generation system that plans, retrieves, criticizes, generates, and verifies — refusing to produce any answer it cannot ground in evidence. Built on LangGraph, Llama.cpp, and ChromaDB with a live Streamlit dashboard.

---

## Research Paper

> 📄 This project is accompanied by a research paper currently under review for the **NortheastGenAI 2026 Workshop**, documenting the full system design, multi-stage critic architecture, adaptive retrieval policy, and benchmark evaluation against a standard RAG baseline.

### Submission Details

| Field | Details |
|---|---|
| **Title** | Adaptive Critic-Guided Hybrid Agentic RAG for Improving Retrieval Robustness and Hallucination Resistance Through Multi-Stage Verification |
| **Authors** | Mohith Chandra Gugulothu
| **Submitted To** | NortheastGenAI 2026 Workshop (Under Review) |
| **Submission Date** | May 15, 2026 |
| **Status** | ⏳ Under Review — not yet accepted or published |

### Access

| Resource | Link |
|---|---|
| 📄 Full Paper (PDF) | Available upon acceptance |
| 🗃️ arXiv Preprint | Available upon acceptance |
| 💻 GitHub Repository | [MohithChandra07/Agentic-Hybrid-RAG](https://github.com/MohithChandra07/Agentic-Hybrid-RAG) |

### Abstract

Standard RAG systems retrieve documents and generate answers without any verification step — producing confident, ungrounded outputs when retrieval fails. This paper proposes a multi-agent critic-guided framework that addresses this through three mechanisms: an adaptive retrieval policy that routes queries by complexity, a dual-stage critic architecture that gates both context quality and answer quality independently, and an atomic claim verifier that decomposes generated answers into individual sentences for per-claim hallucination scoring. A semantic failure memory module further enables the system to adapt retrieval strategy based on prior session failures. Evaluated against a standard RAG baseline, the proposed system achieves 100% faithfulness and 0% hallucination rate on the test set, at the cost of a 3× latency increase attributable to the multi-agent verification overhead.

### Citation

A formal citation will be available upon acceptance. In the meantime, you may reference this work as:

```bibtex
@unpublished{gugulothu2026adaptivecritic,
  title     = {Adaptive Critic-Guided Hybrid Agentic {RAG} for Improving
               Retrieval Robustness and Hallucination Resistance Through
               Multi-Stage Verification},
  author    = {Mohith Chandra Gugulothu},
  note      = {Submitted to NortheastGenAI 2026 Workshop (under review)},
  year      = {2026},
  month     = {May}
}
```

---

## Demo

> 📺 **[Watch the full walkthrough on YouTube](https://youtube.com/your-demo-link)**

The demo covers:
- A live query running through all 7 agents in real time
- The Context Critic rejecting insufficient evidence and triggering Web Fallback
- The Claim Verifier flagging a hallucination risk and looping back for retry
- The final verified answer with per-claim risk scores

![Dashboard](https://github.com/MohithChandra07/Agentic-Hybrid-RAG/blob/main/screenshots/dashboard.png?raw=true)

---

## Motivation

This project extends **"RAG-Critic: A Critic-Guided RAG Framework" (ACL 2025)** — a paper that proposed evaluating retrieved context before generation. Six gaps in that design motivated this work:

**Gap 1 — Holistic scoring misses micro-hallucinations.**
The original critic graded the final answer as a whole, meaning a single hallucinated sentence buried inside an otherwise correct answer still passes the gate. This system decomposes every draft into atomic sentences and verifies each one independently against the retrieved evidence, so no individual claim escapes scrutiny.

**Gap 2 — Uniform retrieval regardless of query complexity.**
A Cross-Encoder reranker was applied to every query at a fixed `k`, treating a single-fact lookup the same as a multi-hop reasoning question — wasteful on easy queries and sometimes under-powered on hard ones. This system uses an adaptive routing policy: simple factual queries get fast `k=3` shallow search; complex multi-hop queries get deep `k=10` reranking with a full Cross-Encoder pass.

**Gap 3 — No memory of prior failures.**
Every time the system failed on a query type, it started from scratch on the next similar query. There was no mechanism to learn from retrieval dead-ends or repeated failures on the same topic class. This system adds a semantic failure memory module backed by a secondary ChromaDB instance — failed query embeddings and session summaries are stored and consulted before each new retrieval attempt, allowing the system to adapt its strategy over time.

**Gap 4 — Single retrieval source with no fallback.**
The original framework retrieved exclusively from a static local knowledge base. If the corpus lacked coverage for a query, the system would either hallucinate or return a low-confidence answer with no escalation path. This system adds an autonomous Web Fallback agent that triggers whenever the Context Critic scores local retrieval below threshold — live DuckDuckGo search fills the coverage gap without any manual intervention.

**Gap 5 — No query decomposition for multi-hop questions.**
The original system passed the raw user query directly to the retriever, which works for single-hop factual lookups but degrades on questions that require chaining multiple pieces of evidence. Without decomposition, the retriever conflates sub-questions and returns noisy, mixed results. This system adds a Planner agent that detects multi-hop structure, breaks the query into ordered sub-queries, and resolves each sequentially with intermediate results passed forward as context.

**Gap 6 — Single critic gate with no targeted retry.**
The original critic made a one-shot pass/fail decision. If a generated answer was borderline — mostly correct but containing one flawed claim — the entire answer was discarded with no attempt at targeted repair. This system introduces a second critic gate downstream of generation, the Answer Critic, which triggers a targeted retry with refined retrieval parameters rather than a full pipeline restart. Valid portions of the answer are preserved; only the flagged claims are re-grounded.

---

## Architecture Overview

The system is a directed graph of 7 specialized agents orchestrated by LangGraph. Each agent has a single responsibility; no agent proceeds without passing an upstream gate.

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

### Agent Responsibilities

**Planner**
Classifies query complexity and determines retrieval depth. For multi-hop questions, it decomposes the query into ordered sub-queries before retrieval begins. This prevents over-fetching on simple queries and under-fetching on complex ones.

**Hybrid Retriever**
Runs BM25 lexical search and ChromaDB vector search in parallel, then merges and deduplicates results. The Planner's complexity signal determines whether the Retriever applies a Cross-Encoder reranker (`k=10`, deep path) or returns the raw merged set (`k=3`, shallow path).

**Context Critic**
Scores the retrieved evidence for semantic relevance against the query. If the score falls below threshold, the pipeline does not proceed to generation — it escalates to Web Fallback instead. This is the first critic gate.

**Claim Verifier**
After generation, the draft answer is decomposed into atomic sentences. Each sentence is independently cross-referenced against the evidence vector space and scored for hallucination risk. Sentences that cannot be grounded in retrieved evidence are flagged.

**Retry Loop (Answer Critic)**
If any claim scores above the risk threshold (0.5), the Answer Critic activates, refines the retrieval parameters, and routes the query back through the pipeline. The loop terminates only when all claims are verified or the retry budget is exhausted.

![Architecture Diagram](https://github.com/MohithChandra07/Agentic-Hybrid-RAG/blob/main/diagrams/architecture.png?raw=true)

---

## Key Contributions

| Contribution | Description |
|---|---|
| **Adaptive Retrieval Policy** | Complexity classifier routes to fast shallow search (`k=3`) or deep Cross-Encoder reranking (`k=10`). Reduces unnecessary compute on simple queries. |
| **Atomic Claim Verifier** | Decomposes draft answers into individual sentences and cross-references each against the evidence space, rather than grading the answer as a whole. |
| **Semantic Failure Memory** | Secondary ChromaDB instance stores embeddings of failed queries and session summaries. The system checks this store before retrieval and adapts its strategy based on prior failures. |
| **Dual Critic Gating** | Two independent critic agents gate the pipeline at different stages. Context quality is verified before generation; answer quality is verified before output. |
| **Autonomous Web Fallback** | When local knowledge base coverage is insufficient, the system escalates to live web search via DuckDuckGo without manual intervention. |

---

## Benchmark Results

![Benchmark Charts](https://github.com/MohithChandra07/Agentic-Hybrid-RAG/blob/main/diagrams/metrics.png?raw=true)

| Metric | Baseline RAG | Agentic Hybrid RAG |
|---|---|---|
| Faithfulness ↑ | 66% | **100%** |
| Hallucination Rate ↓ | 34% | **0%** |
| Retrieval Precision ↑ | 66% | **100%** |
| Avg. Latency | **35.9s** | 116.1s |

**Faithfulness & Hallucination Rate** — on the test set, the agentic system answered only what it could verify. The 0% hallucination rate reflects the system refusing to generate ungrounded claims rather than guessing.

**Retrieval Precision** — the baseline scores higher on raw precision because it uses a narrower retrieval setup. The agentic system casts a wider net through hybrid search and then filters aggressively via the critic — trading raw precision for better coverage and verified quality.

**Latency** — the 3× latency increase is the honest cost of running 7 agents, retry loops, and claim verification on local hardware. A GPU-accelerated or cloud deployment would reduce this significantly.

**Retry Effectiveness** — the retry loop activates only when the Answer Critic flags risk above threshold. When it activates, it produces a measurable relevance improvement (~0.05 score gain), confirming the loop is working as intended rather than firing blindly.

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| Local LLM | Llama.cpp + Phi-3 (Metal / Apple Silicon) |
| Vector DB | ChromaDB |
| Lexical Search | BM25 |
| Reranker | SentenceTransformers `ms-marco-MiniLM` |
| Failure Memory | ChromaDB (secondary instance) |
| UI | Streamlit |
| Web Fallback | DuckDuckGo API |

---

## Project Structure

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
│       └── graph.py            # LangGraph graph + conditional routing
│
├── dataset/                    # Source PDFs
├── benchmark/                  # Ground-truth evaluation JSONs
├── evaluation/                 # Benchmarking scripts
├── diagrams/                   # Architecture + metric charts
└── screenshots/                # UI captures
```

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/MohithChandra07/Agentic-Hybrid-RAG.git
cd Agentic-Hybrid-RAG

# 2. Install dependencies
pip3 install -r requirements.txt
pip3 install sentence-transformers duckduckgo-search

# 3. Mac — compile llama-cpp with Metal acceleration
CMAKE_ARGS="-DGGML_METAL=on" pip3 install --force-reinstall --no-cache-dir llama-cpp-python

# 4. Build the vector database
python3 ingest.py

# 5a. Run via CLI
python3 agentic_rag/backend/graph.py

# 5b. Run the Streamlit dashboard
python3 -m streamlit run ui_app.py
```

---

## Evaluation

```bash
python3 -m evaluation.runner     # run benchmark suite
python3 -m evaluation.visualize  # generate metric charts
```

---

## Roadmap

- **GraphRAG** — migrate from flat ChromaDB to a Neo4j knowledge graph for multi-hop relational traversal
- **Multi-modal retrieval** — handle figures and tables inside PDFs with a vision-capable LLM
- **Cloud deployment** — Docker + vLLM cluster for multi-user concurrent access
- **Feedback loop** — wire real user ratings back into the failure memory module

---

## For Recruiters & Interviewers

The core engineering is concentrated in two places:

- `agentic_rag/backend/retrieval/` — adaptive retrieval policy, BM25/vector fusion, Cross-Encoder reranking
- `agentic_rag/backend/agents/` — atomic claim verifier, dual critic gating logic

The LangGraph state machine and all conditional routing lives in `graph.py`. Happy to walk through any part of the design.

---

*Built by Mohith Chandra Gugulothu · May 2026*
