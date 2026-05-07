
# Conversation Intelligence RAG
### Hierarchical Memory-Augmented Conversational AI System

A production-style conversational memory system that combines:
- Hierarchical RAG
- Topic segmentation
- Persona extraction
- Long-term conversational memory
- FAISS vector retrieval
- Local LLM inference
- Hallucination mitigation

The system is designed to simulate how humans organize memory into:
short-term context, semantic topics, and long-term persona understanding.

---

# 🎯 Project Goal

The primary objective of this project was to build a **Human-Centric Conversational RAG System** capable of remembering, organizing, and retrieving conversational memory over long interactions.

Unlike traditional chatbots that only rely on short context windows, this system introduces a hierarchical memory architecture inspired by human memory systems.

The project specifically focuses on solving two major problems in conversational AI:

## 1. Hallucination

Modern LLMs often invent:
- hobbies
- careers
- relationships
- personality traits
- locations

even when such information was never mentioned.

This project introduces multiple grounding and retrieval validation techniques to reduce unsupported generation.

---

## 2. Context Fragmentation

Traditional RAG pipelines retrieve only small isolated chunks of text.

This causes:
- forgotten topics
- broken conversational continuity
- inconsistent persona understanding

To solve this, the system builds:
- short-term memory
- rolling checkpoint summaries
- long-term semantic topic memory

---

# ✨ Core Features

## 🧠 Hierarchical Conversational Memory

Implements a 3-tier memory system:
- Short-term conversational memory
- Mid-term rolling checkpoints
- Long-term semantic topic memory

---

## 🔍 Semantic Retrieval Pipeline

Uses:
- SentenceTransformers embeddings
- FAISS vector similarity search
- Context-aware retrieval scoring

to retrieve semantically relevant conversation memories.

---

## 🧬 Persona Extraction

Extracts:
- interests
- habits
- personal facts
- communication patterns

while minimizing unsupported inference.

---

## 🚫 Hallucination Mitigation

The system implements multiple grounding safeguards:
- strict system prompts
- retrieval confidence thresholds
- low-temperature deterministic inference
- memory isolation
- retrieval-based refusal handling

---

## 🧩 Topic Segmentation

Automatically detects topic boundaries using:
- semantic embedding similarity
- TextTiling-inspired segmentation
- rolling semantic drift detection

---

## 💻 Fully Local Inference

Runs entirely locally using:
- GGUF models
- local transformers
- CPU-compatible inference pipeline

without relying on external APIs.

---

# 🏗️ System Architecture

The pipeline follows a hierarchical conversational memory architecture:

User Conversation
        ↓
Message Preprocessing
        ↓
Semantic Embedding Generation
        ↓
Topic Segmentation Engine
        ↓
Hierarchical Memory Builder
 ├── Short-Term Memory
 ├── Mid-Term Checkpoints
 └── Long-Term Topic Memory
        ↓
FAISS Vector Indexing
        ↓
Semantic Retrieval
        ↓
Grounded Response Generation
        ↓
Hallucination Validation
        ↓
Final Response

---

# ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Vector Database | FAISS |
| Embeddings | all-MiniLM-L6-v2 |
| LLM | Qwen2.5 / GGUF |
| UI | Streamlit |
| NLP | SentenceTransformers |
| Retrieval | Semantic Vector Search |
| Memory System | Hierarchical RAG |
| Deployment | Local CPU Inference |

---

# 🧠 Memory Architecture

## Tier 1 — Short-Term Memory

Maintains the most recent conversational turns to preserve immediate conversational continuity.

Purpose:
- maintain active context
- support follow-up questions
- improve conversational coherence

---

## Tier 2 — Mid-Term Rolling Checkpoints

Every fixed number of messages are summarized into checkpoint memories.

Purpose:
- compress long conversations
- preserve discussion flow
- capture evolving user context

---

## Tier 3 — Long-Term Topic Memory

A semantic topic segmentation engine groups conversations into long-term thematic memories.

Purpose:
- preserve high-level concepts
- improve topic retrieval
- enable contextual summarization

---

# 🔍 Retrieval Pipeline

The retrieval system combines:
- semantic embeddings
- vector similarity search
- contextual ranking

to identify the most relevant conversational memories.

## Retrieval Flow

1. Convert user query into embedding
2. Search FAISS vector index
3. Retrieve semantically relevant memories
4. Filter low-confidence results
5. Inject grounded memory into generation prompt
6. Generate factual response

---

# 🚫 Hallucination Engineering

One of the primary goals of this project was reducing unsupported memory generation.

The system implements multiple anti-hallucination strategies.

---

## ✅ Strict Grounding

Responses are generated only from retrieved memory fragments.

---

## ✅ Retrieval Confidence Thresholds

If retrieval confidence is weak, the system refuses generation instead of guessing.

Example:
"I do not have enough evidence from stored conversations."

---

## ✅ Persona Isolation

Persona facts are selectively injected instead of globally appended to prompts.

This prevents unrelated persona contamination.

---

## ✅ Deterministic Inference

Low-temperature generation minimizes creative drift.

---

## ✅ Memory Validation

Responses are constrained to retrieved conversational evidence whenever possible.

---

## ✅ Context Segmentation

The memory pipeline separates:
- session memory
- topic memory
- persona memory

to avoid recursive hallucination contamination.

---

# 🚧 Engineering Challenges

## 1. Recursive Hallucination Contamination

Early versions stored AI-generated responses back into memory.

This caused:
- hallucinated facts becoming permanent memory
- recursive misinformation amplification

### Solution

Only user messages are persisted into long-term memory.

---

## 2. Weak Small-Model Grounding

Smaller local models frequently attempted to infer unsupported:
- hobbies
- careers
- relationships
- personality traits

### Solution

Implemented:
- strict prompting
- deterministic decoding
- retrieval confidence rejection
- persona filtering

---

## 3. Topic Drift

Long conversations caused semantic overlap between unrelated topics.

### Solution

Introduced:
- hierarchical topic segmentation
- rolling checkpoints
- semantic smoothing

---

## 4. JSON Instability

Smaller local models often generated malformed structured outputs.

### Solution

Implemented:
- regex recovery pipeline
- multi-stage JSON sanitization
- validation fallback logic

---

# 📊 Performance Characteristics

| Metric | Approximate Value |
|---|---|
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Search | FAISS Flat Index |
| Retrieval Latency | ~50-150ms |
| CPU Inference Speed | ~5-10 tokens/sec |
| Model Size | 1.5B - 3B |
| Context Architecture | 3-Tier Hierarchical Memory |

The system is optimized for:
- local execution
- low-resource hardware
- retrieval-grounded responses

---

# 🎥 Demo & System Behavior

## ✅ Grounded Topic Segmentation & Conversational Memory

![Demo 1](images/demo1.png)

This demo shows the system successfully:
- remembering long-term conversational facts
- organizing discussions into semantic topics
- retrieving grounded contextual memories
- summarizing conversations without losing topic continuity

The chatbot correctly recalls:
- location information
- study habits
- reading preferences

while maintaining conversational flow.

---

## ⚠️ Hallucination Edge Cases & Ongoing Grounding Improvements

![Demo 2](images/demo2.png)

This example demonstrates one of the core engineering challenges in local LLM systems:
small-model hallucination during persona summarization.

Although the retrieval system correctly grounds most information,
smaller local models may still occasionally generate unsupported entities during open-ended summarization tasks.

This project actively mitigates such issues through:
- retrieval confidence filtering
- deterministic generation
- strict grounding prompts
- memory isolation strategies

The goal is not only building a chatbot,
but engineering a robust long-term conversational memory system with practical hallucination reduction techniques.

---

# 🚀 Installation & Setup

## Clone Repository

```bash
git clone <your-repo-link>
cd conversation-intelligence-rag
````

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 📂 Project Structure

```bash
conversation-intelligence-rag/
│
├── artifacts/
│   ├── persona.json
│   ├── vector_meta.json
│   ├── index_topics.faiss
│   ├── index_chunks.faiss
│   └── index_100.faiss
│
├── src/
│   ├── chatbot/
│   ├── retrieval/
│   ├── segmentation/
│   ├── persona/
│   ├── utils/
│   └── ingestion/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🔮 Future Improvements

* Hybrid Retrieval (BM25 + Semantic Search)
* GPU acceleration
* Multi-user memory support
* Memory aging & decay simulation
* Retrieval citation system
* Structured knowledge graphs
* Persistent cloud vector database
* Real-time conversation ingestion
* Streaming response generation
* Multi-modal memory support

---

# 🧠 Key Learning Outcomes

This project explored:

* practical RAG engineering
* hierarchical memory systems
* semantic retrieval pipelines
* hallucination mitigation
* local LLM orchestration
* conversational context management
* production-style AI system design

The system reflects a strong focus on grounded conversational intelligence rather than simple chatbot generation.

---

# 📌 Conclusion

Conversation Intelligence RAG demonstrates how conversational AI systems can move beyond simple context windows toward structured long-term memory architectures.

The project emphasizes:

* grounded retrieval
* memory organization
* hallucination reduction
* engineering reliability

while operating entirely through local inference pipelines.

This project represents a practical exploration into building scalable, memory-aware conversational AI systems using modern RAG engineering principles.

```
```
