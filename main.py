import json
import os
import faiss
import logging
import argparse
from src.ingestion.loader import load_conversations
from src.segmentation.detector import TopicDetector
from src.checkpoints.generator import CheckpointGenerator
from src.persona.extractor import PersonaExtractor
from src.retrieval.index import VectorStore
from src.utils.models import LocalGenerator

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(force_reprocess: bool = False, limit: int = 200):
    """
    Orchestrates the data ingestion and indexing pipeline.
    Implements idempotency to skip redundant heavy computations.
    """
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Check for existing artifacts to enable fast-start
    if not force_reprocess and os.path.exists(f"{artifacts_dir}/vector_meta.json"):
        logger.info("Artifacts already exist. Skipping heavy pipeline. Use --force to reprocess.")
        return

    logger.info(f"Starting pipeline (limit={limit} messages)...")
    
    # 1. Load Data
    messages = load_conversations("data/conversations.csv", limit=limit)
    if not messages:
        logger.error("No messages found in dataset.")
        return
    
    # 2. Initialize Models
    # singleton handles model loading
    generator = LocalGenerator() 
    detector = TopicDetector("all-MiniLM-L6-v2", threshold=0.6, window_size=5)
    checkpoint_gen = CheckpointGenerator(generator)
    persona_ext = PersonaExtractor(generator)
    vector_store = VectorStore("all-MiniLM-L6-v2")

    # 3. Topic Segmentation & Summarization
    logger.info("Segmenting topics and generating summaries...")
    segments = detector.get_segments(messages)
    topic_checkpoints = []
    
    for i, (start, end) in enumerate(segments):
        seg_msgs = messages[start:end+1]
        logger.info(f"Processing Topic {i} ({len(seg_msgs)} msgs)...")
        
        # Generate semantic summary
        chk = checkpoint_gen.generate_topic_checkpoint(i, seg_msgs)
        topic_checkpoints.append(chk.model_dump())
        vector_store.add_topic(chk.model_dump())
        
        # Store raw text chunks for granular RAG
        chunk_text = "\n".join([f"{m.speaker}: {m.text}" for m in seg_msgs])
        vector_store.add_chunk({"text": chunk_text, "start": start, "end": end})

    # 4. Rolling Checkpoints (Mental Context)
    logger.info("Generating rolling 100-message checkpoints...")
    checkpoints_100 = []
    for i in range(0, len(messages), 100):
        block = messages[i:i+100]
        chk = checkpoint_gen.generate_100_checkpoint(i//100, block)
        checkpoints_100.append(chk.model_dump())
        vector_store.add_100(chk.model_dump())

    # 5. Persona Profile Extraction
    logger.info("Performing Persona Analysis...")
    # Assuming "User 1" is the primary subject
    persona = persona_ext.extract(messages, target_speaker="User 1")
    
    # 6. Persistence Layer
    logger.info("Persisting indices and metadata to disk...")
    
    with open(f"{artifacts_dir}/topic_checkpoints.json", "w") as f:
        json.dump(topic_checkpoints, f, indent=2)
        
    with open(f"{artifacts_dir}/100_checkpoints.json", "w") as f:
        json.dump(checkpoints_100, f, indent=2)
        
    with open(f"{artifacts_dir}/persona.json", "w") as f:
        json.dump(persona.model_dump(), f, indent=2)

    faiss.write_index(vector_store.index_topics, f"{artifacts_dir}/index_topics.faiss")
    faiss.write_index(vector_store.index_100, f"{artifacts_dir}/index_100.faiss")
    faiss.write_index(vector_store.index_chunks, f"{artifacts_dir}/index_chunks.faiss")
    
    with open(f"{artifacts_dir}/vector_meta.json", "w") as f:
        json.dump({
            "topics": vector_store.meta_topics,
            "100": vector_store.meta_100,
            "chunks": vector_store.meta_chunks
        }, f, indent=2)
        
    logger.info("Pipeline completed successfully. Deployment artifacts ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Conversation Analysis Pipeline")
    parser.add_argument("--force", action="store_true", help="Force re-processing of all data")
    parser.add_argument("--limit", type=int, default=200, help="Maximum messages to process")
    args = parser.parse_args()
    
    run_pipeline(force_reprocess=args.force, limit=args.limit)
