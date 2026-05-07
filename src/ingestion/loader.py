import pandas as pd
from typing import List
import logging
from src.utils.schemas import Message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_conversations(csv_path: str, limit: int = None) -> List[Message]:
    """
    Load the CSV of conversations and build a canonical message stream.
    Each row is a conversation string where messages are separated by \n.
    """
    logger.info(f"Loading conversations from {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    
    if limit:
        df = df.head(limit)

    messages = []
    message_id = 0
    for _, row in df.iterrows():
        conversation_text = str(row[0])
        lines = conversation_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(":", 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
            else:
                speaker = "Unknown"
                text = line
            
            if text:
                messages.append(Message(
                    message_id=message_id,
                    speaker=speaker,
                    text=text
                ))
                message_id += 1
                
    logger.info(f"Loaded {len(messages)} messages.")
    return messages

if __name__ == "__main__":
    msgs = load_conversations("../../data/conversations.csv", limit=5)
    print(msgs[:3])
