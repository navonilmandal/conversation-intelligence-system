"""
FIXED extractor.py
------------------
BUG (Root cause of hallucination): The extraction prompt said
"DO NOT assume hobbies or jobs" but didn't explicitly forbid
inferring from context. The 3B model still inferred things like
"fulltime student studying radiology" from indirect conversation cues.

FIX: Added a concrete FORBIDDEN EXAMPLES block to the prompt —
the same technique used in service.py's system prompt.
Also added a second-pass deduplication that strips facts which
are clearly inferences (contain words like "seems", "appears",
"probably", "likely", "must be") rather than explicit statements.
"""

import re
import json
import logging
from typing import List, Dict, Any
from src.utils.schemas import Message, Persona, PersonaFact
from src.utils.models import LocalGenerator

logger = logging.getLogger(__name__)

# Words that indicate inference rather than explicit statement
_INFERENCE_MARKERS = {
    "seems", "appears", "probably", "likely", "must be", "must have",
    "presumably", "apparently", "suggests", "implies", "inferred",
    "perhaps", "maybe", "possibly", "could be", "might be"
}


class PersonaExtractor:
    """
    Two-pass Persona Profiling System.
    Pass 1: LLM extraction with strict grounding prompts.
    Pass 2: Deduplication + inference filter.
    """

    def __init__(self, generator: LocalGenerator):
        self.generator = generator

    def _clean_json(self, text: str) -> str:
        """Strip markdown fences and extract raw JSON."""
        text = re.sub(r"```(json)?", "", text)
        text = text.split("```")[0]
        # Find the outermost { } block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group() if match else text.strip()

    def _is_inference(self, fact: str) -> bool:
        """Return True if the fact looks like a model inference, not an explicit statement."""
        fact_lower = fact.lower()
        return any(marker in fact_lower for marker in _INFERENCE_MARKERS)

    def extract(self, messages: List[Message], target_speaker: str = "User 1") -> Persona:
        logger.info(f"Extracting persona for '{target_speaker}'...")

        # Collect only the target speaker's messages
        speaker_messages = [m for m in messages if m.speaker == target_speaker]
        if not speaker_messages:
            logger.warning(f"No messages found for speaker '{target_speaker}'.")
            return Persona()

        evidence_text = [m.text for m in speaker_messages]
        chunk_size = 40   # slightly smaller chunks = better focus for 3B model
        raw_insights = []

        for i in range(0, len(evidence_text), chunk_size):
            chunk = evidence_text[i:i + chunk_size]
            chunk_text = "\n".join(f"- {t}" for t in chunk)

            prompt = f"""You are extracting ONLY explicitly stated facts from a person's messages.

MESSAGES FROM {target_speaker}:
{chunk_text}

STRICT EXTRACTION RULES:
1. A fact is ONLY valid if the person DIRECTLY STATED it in their own words.
2. DO NOT infer, deduce, or guess anything.
3. DO NOT add context you think is likely (e.g. if they mention a hospital, do NOT write "works at a hospital").

FORBIDDEN EXAMPLES (never write things like these):
- "fulltime student studying radiology" — unless they literally said those exact words
- "works in healthcare" — unless they literally said that
- "seems introverted" — inference, not a fact
- "probably likes sports" — inference, not a fact

ALLOWED EXAMPLES (only write things like these):
- "Said they wake up at 6am every day" — explicitly stated
- "Mentioned they have a dog named Luna" — explicitly stated
- "Uses short sentences and rarely uses punctuation" — directly observable

If a category has zero explicit evidence, return an empty list [].

Respond ONLY with this JSON, no other text:
{{
  "habits": [],
  "personal_facts": [],
  "personality_traits": [],
  "communication_style": []
}}"""

            response = self.generator.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a strict factual data extractor. "
                    "Output ONLY valid JSON. "
                    "If something is not explicitly stated word-for-word, it does not belong in the output."
                ),
                max_new_tokens=500,
                temperature=0.0,   # fully deterministic for extraction
            )

            try:
                data = json.loads(self._clean_json(response))
                raw_insights.append(data)
                logger.debug(f"Chunk {i}: extracted {sum(len(v) for v in data.values())} facts")
            except Exception as e:
                logger.error(f"JSON parse failed for chunk {i}: {e}\nRaw: {response[:200]}")

        # Consolidation + deduplication + inference filter
        final_persona = Persona()
        seen_facts: set = set()

        mapping = {
            "habits":             final_persona.habits,
            "personal_facts":     final_persona.personal_facts,
            "personality_traits": final_persona.personality_traits,
            "communication_style":final_persona.communication_style,
        }

        skipped_inference = 0
        for insight in raw_insights:
            for key, target_list in mapping.items():
                for fact in insight.get(key, []):
                    if not isinstance(fact, str):
                        logger.warning(f"Skipping non-string in '{key}': {fact}")
                        continue

                    fact_clean = fact.strip()
                    fact_key   = fact_clean.lower()

                    if not fact_key or fact_key in seen_facts:
                        continue

                    # Drop inferred facts
                    if self._is_inference(fact_clean):
                        logger.debug(f"Dropped inference: {fact_clean}")
                        skipped_inference += 1
                        continue

                    seen_facts.add(fact_key)
                    target_list.append(PersonaFact(
                        fact=fact_clean,
                        evidence_message_ids=[],
                        confidence=0.9,
                    ))

        logger.info(
            f"Persona built: {len(seen_facts)} facts kept, "
            f"{skipped_inference} inferences dropped."
        )
        return final_persona