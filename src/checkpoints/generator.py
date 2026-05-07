"""
FIXED generator.py
------------------
BUG 1: _safe_json_parse fallback did text.split("summary")[-1] which
       splits on ANY occurrence of the word "summary" inside a sentence
       and produces garbage like ": Student discussed their radiology course."
       Fixed: fallback now returns a safe generic string, not a slice of raw text.

BUG 2: topic_label prompt had no example of what a BAD label looks like,
       so the 3B model would write inferred category names like
       "Academic Life" or "Career Discussion" based on one mention of a subject.
       Fixed: added FORBIDDEN label examples to the prompt.

BUG 3: 100-checkpoint prompt had no system grounding at all for the 3B model —
       it would freely summarize with inferences. Fixed with explicit rules.
"""

import json
import logging
import re
from typing import List, Dict, Any
from src.utils.schemas import Message, TopicSegment, Checkpoint100
from src.utils.models import LocalGenerator

logger = logging.getLogger(__name__)


class CheckpointGenerator:
    """
    Semantic Summarization Engine.
    Generates structured topic labels and rolling checkpoints for long-term memory.
    """

    def __init__(self, generator: LocalGenerator):
        self.generator = generator

    def _format_conversation(self, messages: List[Message]) -> str:
        return "\n".join(f"{m.speaker}: {m.text}" for m in messages)

    def _safe_json_parse(self, text: str, default_val: Dict[str, Any]) -> Dict[str, Any]:
        """
        Robustly extract JSON from model output.
        FIX BUG 1: Fallback returns safe defaults, not a slice of raw text.
        """
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(text)
        except Exception:
            logger.warning(f"JSON parse failed. Raw output: {text[:150]}")
            # FIX: return clean defaults, not a corrupted text slice
            return default_val

    def generate_topic_checkpoint(self, segment_id: int, messages: List[Message]) -> TopicSegment:
        """Generate a factual summary for a detected topic segment."""
        conv_text = self._format_conversation(messages[:50])

        prompt = f"""Summarize ONLY what is explicitly discussed in this conversation segment.

Conversation:
{conv_text}

RULES:
1. topic_label must be a SHORT phrase (2-4 words) taken literally from the conversation.
2. summary must be ONE sentence describing only what was directly said.
3. keywords must be words that actually appear in the text.

FORBIDDEN topic_label examples (do NOT write vague inferred categories):
- "Academic Life" (too inferred)
- "Career Discussion" (too inferred)  
- "Personal Background" (too vague)

ALLOWED topic_label examples (literal and specific):
- "Dog Walking Routine"
- "Weekend Plans"
- "Cooking Preferences"

Respond ONLY with valid JSON, no other text:
{{
  "summary": "one literal sentence",
  "topic_label": "short literal label",
  "keywords": ["word1", "word2", "word3"]
}}"""

        response = self.generator.generate(
            prompt=prompt,
            system_prompt="You are a data labeling assistant. Output ONLY valid JSON. Never infer or guess.",
            max_new_tokens=200,
            temperature=0.0,
        )

        data = self._safe_json_parse(
            response,
            {"summary": "Conversation segment.", "topic_label": "General", "keywords": []}
        )

        return TopicSegment(
            segment_id=segment_id,
            start_message_id=messages[0].message_id,
            end_message_id=messages[-1].message_id,
            topic_label=str(data.get("topic_label", "General"))[:60],   # cap length
            summary=str(data.get("summary", ""))[:300],
            keywords=data.get("keywords", [])[:10],
        )

    def generate_100_checkpoint(self, checkpoint_id: int, messages: List[Message]) -> Checkpoint100:
        """Generate a rolling summary for a 100-message block."""
        context_slice = messages[:20] + messages[-20:]
        conv_text = self._format_conversation(context_slice)

        prompt = f"""These are sampled messages from a 100-message conversation block.
Write a factual summary of ONLY what was explicitly discussed.
Do NOT infer topics, jobs, or personal details not directly stated.

Messages:
{conv_text}

Respond ONLY with valid JSON:
{{
  "summary": "Factual one-sentence overview of what was discussed",
  "keywords": ["topic1", "topic2", "topic3"]
}}"""

        response = self.generator.generate(
            prompt=prompt,
            system_prompt="You are a concise factual summarizer. Output ONLY valid JSON. No inference.",
            max_new_tokens=200,
            temperature=0.0,
        )

        data = self._safe_json_parse(
            response,
            {"summary": "Conversation block.", "keywords": []}
        )

        return Checkpoint100(
            checkpoint_id=checkpoint_id,
            start_message_id=messages[0].message_id,
            end_message_id=messages[-1].message_id,
            summary=str(data.get("summary", ""))[:300],
            keywords=data.get("keywords", [])[:10],
        )