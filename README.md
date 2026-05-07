# Conversation Intelligence: Hierarchical RAG & Zero-Hallucination Engineering

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![GGUF Inference](https://img.shields.io/badge/Inference-GGUF%20CPU-orange.svg)](https://github.com/ggerganov/llama.cpp)
[![Memory Architecture](https://img.shields.io/badge/Memory-Hierarchical%203--Tier-green.svg)](https://github.com/facebookresearch/faiss)

## 🎯 The Goal
The primary objective of this project was to build a **Human-Centric RAG Chatbot** that actually remembers past conversations without the "Memory Decay" common in standard LLMs. We aimed to solve the two biggest failures in modern RAG:
1. **Hallucination**: The model inventing personal details (hobbies, jobs) not present in the data.
2. **Context Fragmentation**: The model "forgetting" high-level topics because it only looks at tiny text snippets.

## 🧠 How It Works (The Architecture)
Unlike standard RAG which only uses raw text chunks, this system implements a **Hierarchical Memory Pipeline**:

1. **Tier 1: Short-term (Session History)**: Keeps the last 20 turns in a structured dictionary to maintain immediate conversational flow.
2. **Tier 2: Mid-term (Rolling Checkpoints)**: Summarizes every 100 messages into "Mental Snapshots" to capture shifts in user sentiment.
3. **Tier 3: Long-term (Topic Segmentation)**: Uses a semantic TextTiling-inspired algorithm to detect topic shifts and index them as high-level summaries.

## 🛠️ How We Solved the "Hallucination Problem"
We moved away from standard prompting and implemented **"Nuclear Grounding"**:
- **Role Locking**: We restructured the system prompt to explicitly define the AI as an "Assistant" and the memory as "User Facts." This stopped the AI from adopting the user's identity.
- **Negative Constraints**: We added "Forbidden Examples" to the prompt, explicitly telling the model *not* to guess hobbies like "Scuba Diving" or "Radiology" just because it saw a hospital mentioned.
- **Deterministic Extraction**: The Persona Extractor uses a high-token window and a second-pass inference filter to strip out "guesses" like "seems like a student" while keeping facts like "is a student."

## 🚧 Challenges & "War Stories" (Development Journey)
During development, we faced several critical engineering hurdles:
- **The "Radiology Student" Bug**: Early versions suffered from "Identity Injection," where the 3B model would see a user's past mention of a radiology course and start introducing itself as a radiology student. We fixed this by strictly isolating the memory block from the AI's persona.
- **The JSON Corruption Crisis**: Small models (0.5B - 3B) often struggle to output valid JSON. We built a **Robust JSON Recovery Engine** using RegEx and recursive parsing to "save" malformed responses.
- **The 2GB Git Bloat**: We accidentally committed a 2GB GGUF model to history, which taught us the vital importance of a robust `.gitignore` and Git history scrubbing before the final push.

## ⚠️ The "Bad Sides" (Honest Limitations)
Every engineering system has trade-offs. The current version has these limitations:
1. **Cold-Boot Latency**: Loading the 2.1GB GGUF model and initializing the FAISS indices on a CPU can take 10-15 seconds.
2. **CPU Inference Speed**: While highly optimized, response generation is slower than a GPU-hosted API (approx. 5-10 tokens/sec).
3. **Small-Model Reasoning**: While Qwen2.5-3B is powerful, it lacks the massive world knowledge of GPT-4, meaning it relies 100% on the quality of your retrieved context.

## 📈 The Roadmap (Overcoming Limitations)
- **Hybrid Search**: Implementing BM25 (keyword) alongside Semantic (vector) search to catch specific names or dates that vectors might miss.
- **GPU Acceleration**: Migrating the `llama-cpp-python` backend to use CUDA for near-instant responses.
- **Multi-User Scaling**: Moving from a local FAISS flat-file to a hosted Vector Database (like Pinecone or Milvus) for production scale.

---
*This project represents a deep dive into the practical realities of local LLM orchestration and data-grounding engineering.*
