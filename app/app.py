"""
app.py — GGUF Edition (Streamlit)
----------------------------------
LocalGenerator now loads the GGUF file automatically.
No MODEL_ID string — just run download_model.py first.
"""

import streamlit as st
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.models import LocalGenerator
from src.chatbot.service import ChatbotService
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Conversation Intelligence", layout="wide")
st.title("Conversation Intelligence Chatbot")
st.markdown("Ask questions about the user's persona, topics, or conversation details.")

try:
    if "generator" not in st.session_state:
        with st.spinner("Loading GGUF model (~2 GB, one-time load)..."):
            st.session_state.generator = LocalGenerator()

    if "embedder" not in st.session_state:
        st.session_state.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    if "chatbot" not in st.session_state:
        artifacts_path = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        st.session_state.chatbot = ChatbotService(
            st.session_state.generator,
            artifacts_dir=artifacts_path,
            st_model=st.session_state.embedder,
        )
except Exception as e:
    st.error(f"Failed to load: {e}\nMake sure you ran download_model.py and the ingestion pipeline.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., What kind of person is this user?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.answer_question(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})