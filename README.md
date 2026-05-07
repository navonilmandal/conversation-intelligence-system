# Conversation Intelligence: A Hierarchical RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Model](https://img.shields.io/badge/Model-Qwen2.5--3B--Instruct--GGUF-orange.svg)
![Memory](https://img.shields.io/badge/Memory-Hierarchical%20FAISS-green.svg)

A production-grade Retrieval-Augmented Generation (RAG) system designed for absolute factual integrity. This chatbot eliminates LLM hallucinations by enforcing strict evidence-based grounding across a multi-tier memory architecture.

## 🚀 Key Engineering Features

- **GGUF 3B Intelligence Engine**: Leverages `Qwen2.5-3B-Instruct` (Q4_K_M) for high-accuracy reasoning on standard CPU hardware with a minimal 2.1GB RAM footprint.
- **Hierarchical Memory System**: 
  - **Tier 1 (Short-term)**: Stateful session history with structured dictionary anchoring.
  - **Tier 2 (Mid-term)**: 100-message rolling summaries and automated Persona Profiles.
  - **Tier 3 (Long-term)**: Semantic Topic Segmentation for high-level historical recall.
- **Zero-Hallucination Guardrails**: Implements "Nuclear Grounding" via one-shot prompting and negative constraints, forcing the AI to acknowledge insufficient context rather than inventing details.
- **Auditable Retrieval**: Every response is generated via a FAISS vector search (Inner Product on Normalized Vectors) with real-time relevance score logging.

## 🛠️ Technology Stack

- **LLM Engine**: `llama-cpp-python` (GGUF Backend)
- **Embeddings**: `Sentence-Transformers` (`all-MiniLM-L6-v2`)
- **Vector Search**: `FAISS` (Facebook AI Similarity Search)
- **Backend**: `Flask` with Singleton Model Lifecycle Management
- **Frontend**: Responsive SPA with Glassmorphic Design

## 📦 Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```

2. **Download the Brain**:
   ```bash
   python scratch/download_model.py
   ```

3. **Build the Memory (Ingestion)**:
   ```bash
   python main.py --force
   ```

4. **Launch the Intelligence**:
   ```bash
   python app/server.py
   ```

## 🧠 Architectural Overview

The system uses a **TextTiling-inspired semantic segmenter** to detect topic shifts in raw conversation data. These segments are summarized into **Topic Checkpoints** and indexed into a hierarchical FAISS store. During inference, the system performs a multi-stage search to build a "Evidence Context" block, which is then fed into a role-locked system prompt to generate a grounded response.

---
*Developed for Advanced Agentic Coding & Conversation Intelligence Research.*
