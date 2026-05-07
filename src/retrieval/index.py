import numpy as np
import faiss
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Hierarchical FAISS Vector Store.
    Manages Topics, 100-msg blocks, and Raw Chunks with L2-normalized Inner Product (Cosine).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", st_model=None):
        self.model = st_model or SentenceTransformer(model_name)
        self.dim = 384  # MiniLM-L6-v2 dimension
        
        # Initialize empty FlatIP indices (Inner Product on Normalized = Cosine)
        self.index_topics = faiss.IndexFlatIP(self.dim)
        self.index_100 = faiss.IndexFlatIP(self.dim)
        self.index_chunks = faiss.IndexFlatIP(self.dim)
        
        self.meta_topics = []
        self.meta_100 = []
        self.meta_chunks = []

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)

    # --- Indexing Methods ---

    def add_topic(self, topic: Dict[str, Any]):
        text = f"{topic['topic_label']}: {topic['summary']}"
        emb = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        self.index_topics.add(self._normalize(emb))
        self.meta_topics.append(topic)

    def add_100(self, checkpoint: Dict[str, Any]):
        emb = self.model.encode([checkpoint['summary']], convert_to_numpy=True, show_progress_bar=False)
        self.index_100.add(self._normalize(emb))
        self.meta_100.append(checkpoint)

    def add_chunk(self, chunk: Dict[str, Any]):
        emb = self.model.encode([chunk['text']], convert_to_numpy=True, show_progress_bar=False)
        self.index_chunks.add(self._normalize(emb))
        self.meta_chunks.append(chunk)

    # --- Search Method ---

    def search(self, query: str, top_k: int = 5, index_type: str = "chunks") -> List[Dict[str, Any]]:
        """
        Generic search across different hierarchical indices.
        """
        index_map = {
            "topics": (self.index_topics, self.meta_topics),
            "100": (self.index_100, self.meta_100),
            "chunks": (self.index_chunks, self.meta_chunks)
        }
        
        index, meta = index_map.get(index_type, index_map["chunks"])

        if index is None or index.ntotal == 0:
            return []

        emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        norm_emb = self._normalize(emb)
        
        D, I = index.search(norm_emb, min(top_k, index.ntotal))
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(meta):
                hit = meta[idx].copy()
                hit['relevance_score'] = float(score)
                results.append(hit)
        
        return results
