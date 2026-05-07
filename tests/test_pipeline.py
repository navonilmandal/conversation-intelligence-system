import pytest
from src.utils.schemas import Message
from src.segmentation.detector import TopicDetector
from src.checkpoints.generator import CheckpointGenerator

class MockGenerator:
    def __init__(self):
        pass
    def generate(self, prompt, system_prompt="", max_new_tokens=0):
        return '{"summary": "test summary", "topic_label": "test topic", "keywords": ["test"], "habits": [], "personal_facts": [{"fact": "likes test", "evidence_message_ids": [0], "confidence": 0.9}]}'

def test_chronological_ordering():
    messages = [
        Message(message_id=0, speaker="User 1", text="A"),
        Message(message_id=1, speaker="User 2", text="B"),
        Message(message_id=2, speaker="User 1", text="C")
    ]
    assert all(messages[i].message_id < messages[i+1].message_id for i in range(len(messages)-1))

def test_topic_segmentation_boundaries():
    detector = TopicDetector(model_name="all-MiniLM-L6-v2", window_size=1, threshold=0.99)
    messages = [
        Message(message_id=0, speaker="User 1", text="I love apples."),
        Message(message_id=1, speaker="User 2", text="Fruits are great."),
        Message(message_id=2, speaker="User 1", text="Let's talk about quantum physics."),
        Message(message_id=3, speaker="User 2", text="The universe is vast.")
    ]
    segments = detector.get_segments(messages)
    
    last_end = -1
    for start, end in segments:
        assert start == last_end + 1
        assert start <= end
        last_end = end
    assert last_end == len(messages) - 1

def test_100_checkpoint_generation():
    generator = MockGenerator()
    checkpointer = CheckpointGenerator(generator)
    messages = [Message(message_id=i, speaker="User", text=str(i)) for i in range(10)]
    
    chk = checkpointer.generate_100_checkpoint(0, messages)
    assert chk.checkpoint_id == 0
    assert chk.start_message_id == 0
    assert chk.end_message_id == 9
    assert chk.summary == "test summary"
