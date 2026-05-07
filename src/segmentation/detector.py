"""
FIXED detector.py
-----------------
BUG 1: np.convolve with mode='same' zero-pads both ends of the array.
       This creates artificially low scores at position 0 and position n-2,
       which the local-minimum detector will flag as boundaries even if the
       conversation hasn't changed topic at all. Changed to mode='valid' and
       adjusted the index loop accordingly.

BUG 2: main.py passes threshold=0.6 to TopicDetector but the class default
       is 0.45. The inconsistency means if the class is ever instantiated
       without keyword args (e.g., in a test) you get far too many segments.
       Aligned the default to 0.6 to match the intended production value.

BUG 3: When n == window_size * 4 exactly, the guard condition `n < window_size*4`
       falls through and then the coherence_scores loop may produce an empty list,
       causing the final `boundaries.append((start, n-1))` to return a single
       full-conversation segment without error — but also without any detection.
       Changed to `<=` for clarity.
"""

import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import logging
from src.utils.schemas import Message

logger = logging.getLogger(__name__)


class TopicDetector:
    """
    TextTiling-inspired semantic segmentation.
    Detects topic boundaries by finding local minima in inter-window cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.6,          # FIX BUG 2: was 0.45 in original default
        window_size: int = 3,
    ):
        logger.info(f"Initializing TopicDetector (model={model_name}, threshold={threshold})")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.window_size = window_size

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

    def get_segments(self, messages: List[Message]) -> List[Tuple[int, int]]:
        """Return (start_idx, end_idx) pairs for each detected topic segment."""
        if not messages:
            return []

        n = len(messages)
        min_required = self.window_size * 4

        # FIX BUG 3: <= so we don't silently fall through on exact boundary
        if n <= min_required:
            logger.info("Conversation too short for segmentation. Treating as one topic.")
            return [(0, n - 1)]

        texts = [f"{m.speaker}: {m.text}" for m in messages]
        embeddings = self.model.encode(texts, batch_size=32, convert_to_numpy=True)

        # Coherence score at each position i = similarity between left window and right window
        coherence_scores = []
        for i in range(n - 1):
            left_start = max(0, i - self.window_size + 1)
            left_window  = embeddings[left_start : i + 1]
            right_end    = min(n, i + self.window_size + 1)
            right_window = embeddings[i + 1 : right_end]

            sim = self._cosine_similarity(
                left_window.mean(axis=0), right_window.mean(axis=0)
            )
            coherence_scores.append(sim)

        scores_arr = np.array(coherence_scores)

        # FIX BUG 1: Use mode='valid' (kernel=[1/3,1/3,1/3], length 3)
        # This avoids zero-padding artefacts at the edges.
        # 'valid' output length = len(scores_arr) - 2, so valid indices are [1 .. n-3]
        kernel = np.ones(3) / 3
        smoothed = np.convolve(scores_arr, kernel, mode="valid")

        # smoothed[i] corresponds to original index i+1
        boundaries = []
        start = 0
        min_segment_length = self.window_size * 2

        for i, score in enumerate(smoothed):
            orig_idx = i + 1   # offset from 'valid' convolution

            # Local minimum check on smoothed array (needs at least 1 neighbour each side)
            if i == 0 or i == len(smoothed) - 1:
                continue

            is_local_min = smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]
            is_below_threshold = smoothed[i] < self.threshold

            if is_local_min and is_below_threshold:
                if (orig_idx + 1) - start >= min_segment_length:
                    boundaries.append((start, orig_idx))
                    start = orig_idx + 1

        # Always close the final segment
        if start < n:
            boundaries.append((start, n - 1))

        logger.info(f"Detected {len(boundaries)} topic segment(s).")
        return boundaries