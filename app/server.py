"""
server.py — GGUF Edition
-------------------------
No MODEL_ID string anymore — LocalGenerator now resolves the GGUF file path
automatically from: env var GGUF_MODEL_PATH > models/ folder > error.

Run download_model.py before starting this server.
"""

import os
import sys
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Fix: Windows colorama OSError 6 — Flask's colored banner crashes when the
# console handle is invalid (common in Windows terminals and some IDEs).
# These two env vars tell colorama/click to skip ANSI color codes entirely.
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.models import LocalGenerator
from src.chatbot.service import ChatbotService
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 4000

_boot_error = None
chatbot = None

try:
    logger.info("Initializing GGUF LLM Engine...")
    generator = LocalGenerator()   # resolves GGUF path automatically

    logger.info("Initializing Embedding Engine...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    artifacts_path = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    logger.info(f"Loading ChatbotService from: {artifacts_path}")
    chatbot = ChatbotService(generator, artifacts_dir=artifacts_path, st_model=embedder)
    logger.info("All systems ready.")
except Exception as e:
    logger.critical(f"Boot failed: {e}")
    _boot_error = str(e)

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
CORS(app)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if chatbot is None:
        return jsonify({"error": f"System not ready: {_boot_error}"}), 503

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "'question' field required."}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify({"error": f"Exceeds {MAX_QUESTION_CHARS} char limit."}), 400

    try:
        logger.info(f"PROMPT: {question[:80]}...")
        answer = chatbot.answer_question(question)
        logger.info("RESPONSE GENERATED.")
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({"error": "Internal inference error."}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status":       "operational" if chatbot else "degraded",
        "model_loaded": chatbot is not None,
        "engine":       generator.model_id if chatbot else None,
        "device":       generator.device   if chatbot else None,
        "error":        _boot_error,
    })


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  CONVERSATION INTELLIGENCE — GGUF EDITION")
    print("  http://localhost:5000")
    print("═" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)