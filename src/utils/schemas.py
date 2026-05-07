from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    message_id: int
    speaker: str
    text: str

class TopicSegment(BaseModel):
    segment_id: int
    start_message_id: int
    end_message_id: int
    topic_label: str
    summary: str
    keywords: List[str]

class Checkpoint100(BaseModel):
    checkpoint_id: int
    start_message_id: int
    end_message_id: int
    summary: str
    keywords: List[str]

class PersonaFact(BaseModel):
    fact: str
    evidence_message_ids: List[int]
    confidence: float

class Persona(BaseModel):
    habits: List[PersonaFact] = []
    personal_facts: List[PersonaFact] = []
    personality_traits: List[PersonaFact] = []
    communication_style: List[PersonaFact] = []
