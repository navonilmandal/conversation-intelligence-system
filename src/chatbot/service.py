"""
FIXED service.py
----------------
BUG 1 (CRITICAL - Screenshot Issue): The original _get_dynamic_identity() injected persona
       facts into the system prompt under an [IDENTITY_ANCHOR] label. Small models (0.5B-3B)
       interpret this as *their own identity*, causing them to respond IN FIRST PERSON as the
       persona (e.g., "I live in the countryside with my dog Luna") instead of talking ABOUT
       the user. Fixed by restructuring the system prompt so the AI's role is unambiguous.

BUG 2: _load_indices() never loaded index_100.faiss, and _load_metadata() never loaded
       meta_100. 100-message checkpoints were completely ignored during retrieval.

BUG 3: Session history was appended as raw strings. If the model generates a response
       containing "Friend:" or "AI:", the stop-sequence split in answer_question()
       could silently corrupt the history with empty strings.

BUG 4: Memory context was injected even when empty, producing confusing "No relevant
       archives." noise that small models sometimes latch onto and hallucinate from.
"""

import json
import faiss
import logging
import os
from typing import Dict, Any, List, Optional
from src.utils.models import LocalGenerator
from src.retrieval.index import VectorStore

logger = logging.getLogger(__name__)


class ChatbotService:
    """
    Advanced Human-Centric Chatbot with Multi-Tier Memory & Auditable RAG.

    Tier 1 Memory: Stateful Session History (Short-term)
    Tier 2 Memory: Dynamic Persona Profile (Mid-term)
    Tier 3 Memory: FAISS Vector RAG (Long-term, topics + 100-msg + chunks)
    """

    def __init__(self, generator: LocalGenerator, artifacts_dir: str = "artifacts", st_model=None):
        self.generator = generator
        self.artifacts_dir = artifacts_dir
        self.session_history: List[Dict[str, str]] = []   # FIX: structured dicts, not raw strings
        self.persona: Dict[str, Any] = {}

        self.vector_store = VectorStore("all-MiniLM-L6-v2", st_model=st_model)
        self._load_indices()
        self._load_metadata()

    # ------------------------------------------------------------------ #
    #  Persistence Layer                                                   #
    # ------------------------------------------------------------------ #

    def _load_indices(self):
        """FIX BUG 2: Load ALL three FAISS indices, not just two."""
        index_map = {
            "index_topics": "index_topics.faiss",
            "index_100":    "index_100.faiss",      # <-- was missing
            "index_chunks": "index_chunks.faiss",
        }
        for attr, fname in index_map.items():
            path = os.path.join(self.artifacts_dir, fname)
            try:
                setattr(self.vector_store, attr, faiss.read_index(path))
                logger.info(f"Loaded FAISS index: {fname}")
            except Exception as e:
                logger.warning(f"Could not load {fname}: {e}. Index will be empty.")

    def _load_metadata(self):
        """FIX BUG 2: Load ALL metadata keys including '100' checkpoint metas."""
        try:
            meta_path = os.path.join(self.artifacts_dir, "vector_meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.vector_store.meta_topics = meta.get("topics", [])
            self.vector_store.meta_100    = meta.get("100", [])     # <-- was missing
            self.vector_store.meta_chunks = meta.get("chunks", [])
            logger.info("Loaded vector metadata.")
        except Exception as e:
            logger.warning(f"vector_meta.json not found or malformed: {e}")

        try:
            persona_path = os.path.join(self.artifacts_dir, "persona.json")
            with open(persona_path, "r") as f:
                self.persona = json.load(f)
            logger.info("Loaded persona profile.")
        except Exception as e:
            logger.warning(f"persona.json not found: {e}")

    # ------------------------------------------------------------------ #
    #  Prompt Building                                                     #
    # ------------------------------------------------------------------ #

    def _build_persona_summary(self) -> str:
        """
        FIX BUG 1: Returns persona facts as a clearly-labelled third-person
        reference block. Never injects them as the AI's own identity.
        Returns empty string (not a placeholder) if no facts exist, so we can
        skip the block entirely and avoid confusing the model.
        """
        lines = []
        category_labels = {
            "personal_facts":    "Personal facts",
            "habits":            "Habits",
            "personality_traits":"Personality traits",
            "communication_style":"Communication style",
        }
        for key, label in category_labels.items():
            facts = [f["fact"] for f in self.persona.get(key, []) if isinstance(f, dict)]
            if facts:
                lines.append(f"[{label}]")
                lines.extend(f"  - {fact}" for fact in facts)

        return "\n".join(lines)

    def _get_memory_fragments(self, query: str) -> str:
        """
        Multi-tier RAG retrieval with relevance gating.
        Unchanged logic but now also queries index_100 for mid-range context.
        """
        context_parts = []
        q_lower = query.lower()

        # Tier 3a – Persona (always included if available)
        persona_block = self._build_persona_summary()
        if persona_block:
            context_parts.append(f"=== Known facts about the user ===\n{persona_block}")

        summary_keywords = {"what we talked", "summarize", "everything", "history",
                            "recall", "overview", "topics"}

        if any(kw in q_lower for kw in summary_keywords):
            # Full topic dump for summary-style queries
            all_topics = [
                f"  - {t['topic_label']}: {t['summary']}"
                for t in self.vector_store.meta_topics[:12]
            ]
            if all_topics:
                context_parts.append("=== Past discussion topics ===\n" + "\n".join(all_topics))
        else:
            # Semantic topic search
            topic_hits = self.vector_store.search(query, top_k=3, index_type="topics")
            relevant_topics = [h for h in topic_hits if h.get("relevance_score", 0) > 0.45]
            if relevant_topics:
                lines = [f"  - {h['topic_label']}: {h['summary']}" for h in relevant_topics]
                context_parts.append("=== Related past topics ===\n" + "\n".join(lines))

            # 100-message checkpoint search (NEW – was never called before)
            chk_hits = self.vector_store.search(query, top_k=2, index_type="100")
            relevant_chk = [h for h in chk_hits if h.get("relevance_score", 0) > 0.45]
            if relevant_chk:
                lines = [f"  - {h['summary']}" for h in relevant_chk]
                context_parts.append("=== Related period summaries ===\n" + "\n".join(lines))

            # Deep chunk search
            chunk_hits = self.vector_store.search(query, top_k=6, index_type="chunks")
            relevant_chunks = [h for h in chunk_hits if h.get("relevance_score", 0) > 0.52]
            if relevant_chunks:
                lines = [f"  - {h['text'][:200]}" for h in relevant_chunks]
                context_parts.append("=== Relevant conversation excerpts ===\n" + "\n".join(lines))

        return "\n\n".join(context_parts)

    # ------------------------------------------------------------------ #
    #  Core Turn                                                           #
    # ------------------------------------------------------------------ #

    def answer_question(self, question: str) -> str:
        """
        FIX BUG 1 (MAIN FIX): Restructured prompt so the LLM is clearly an
        AI assistant talking *to* the user, not *as* the user's persona.
        FIX BUG 3: History stored as structured dicts; formatted only at
        prompt-build time to avoid corruption.
        """
        # Safety valve: keep last 20 turns (40 entries) for small-model context budget
        if len(self.session_history) > 40:
            logger.warning("Context valve triggered. Truncating to last 20 turns.")
            self.session_history = self.session_history[-40:]

        memory = self._get_memory_fragments(question)

        # Format recent conversation for the prompt
        if self.session_history:
            flow_lines = []
            for turn in self.session_history[-10:]:   # last 5 turns is enough for small models
                flow_lines.append(f"User: {turn['user']}")
                flow_lines.append(f"Assistant: {turn['assistant']}")
            flow = "\n".join(flow_lines)
        else:
            flow = ""   # FIX BUG 4: empty string, not "No prior activity."

        # --- SYSTEM PROMPT (FIX BUG 1) ---
        # The AI is an assistant. Background facts belong to *the user*, not the AI.
        system_prompt = (
            "You are a helpful, warm AI assistant with excellent memory. "
            "You are having a conversation WITH the user. "
            "You remember facts about them from past conversations (shown below), "
            "but you are NOT the user — you are their assistant.\n\n"
            "STRICT RULES:\n"
            "1. ONLY use facts that are explicitly present in the 'Background Memory' block.\n"
            "2. NEVER invent details (hobbies, locations, pets) that are not in memory.\n"
            "3. If the user tells you something new (like 'I live in Kolkata'), "
            "acknowledge it naturally and remember it for this session.\n"
            "4. Respond in 1-3 sentences. Be warm but concise."
        )

        # --- USER PROMPT ---
        # FIX BUG 4: Only include memory block if non-empty
        memory_section = (
            f"--- Background Memory ---\n{memory}\n\n"
            if memory.strip()
            else ""
        )
        conv_section = (
            f"--- Recent Conversation ---\n{flow}\n\n"
            if flow.strip()
            else ""
        )

        full_prompt = (
            f"{memory_section}"
            f"{conv_section}"
            f"User: {question}\n"
            "Assistant:"
        )

        response = self.generator.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_new_tokens=200,
            temperature=0.1,
            stop_sequences=["User:", "Assistant:", "Human:", "---"]
        )

        # Clean trailing artefacts
        for stop in ["User:", "Assistant:", "Human:", "---"]:
            if stop in response:
                response = response.split(stop)[0]
        response = response.strip()

        # FIX BUG 3: Store as structured dict
        self.session_history.append({"user": question, "assistant": response})

        logger.info(
            f"TURN COMPLETE | turns={len(self.session_history)} | "
            f"mem_chars={len(memory)}"
        )
        return response