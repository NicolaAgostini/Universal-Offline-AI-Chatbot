# streamlit_app.py



import streamlit as st
from src.config import DATA_PATH, DB_FAISS_PATH
from src.utils import stylish_heading
from src.loader import load_pdf_files
from src.chunker import create_chunks
from src.embedding import get_embedding_model
from src.vectorstore import build_vector_db, load_vector_db
from src.model_loader import load_llm
from src.prompts import CUSTOM_PROMPT_TEMPLATE, set_custom_prompt
from src.qa_chain import setup_qa_chain
import os

# Page config
st.set_page_config(page_title="Universal AI Chatbot", page_icon="🧠", layout="centered")

# Header
stylish_heading()
st.markdown("<h2 style='text-align: center;'>🧠 Ask Me Anything From Your PDFs</h2>", unsafe_allow_html=True)
st.markdown("✅ Powered by **Offline Mistral** + **FAISS**. Add `.pdf` files in the `/data` folder to change knowledge base.")

st.divider()

# Load pipeline once
@st.cache_resource(show_spinner="Warming up the brain... 🧠⚙️")
def load_pipeline():
    documents = load_pdf_files(DATA_PATH)
    chunks = create_chunks(documents)
    embedding_model = get_embedding_model()

    if not os.path.exists(DB_FAISS_PATH):
        build_vector_db(chunks, embedding_model, DB_FAISS_PATH)

    db = load_vector_db(DB_FAISS_PATH, embedding_model)
    llm = load_llm()
    qa_chain = setup_qa_chain(llm, db, set_custom_prompt(CUSTOM_PROMPT_TEMPLATE))
    return qa_chain

qa_chain = load_pipeline()

# Start chat session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input/output loop
st.chat_message("assistant").markdown("👋 Hello! I'm your AI assistant. Ask me anything about your uploaded documents.")

user_input = st.chat_input("Type your question here...")

if user_input:
    # Store user message
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🧠"):
            try:
                response = qa_chain.invoke({"query": user_input})
                st.markdown(f"🤖 {response['result']}")
                st.session_state.chat_history.append(("bot", response['result']))
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Optionally: show previous messages
if st.session_state.chat_history:
    st.divider()
    st.markdown("🕓 **Conversation History**", help="Scroll back through your past questions and answers.")
    for sender, msg in st.session_state.chat_history:
        icon = "🧠" if sender == "user" else "🤖"
        st.markdown(f"**{icon} {sender.capitalize()}**: {msg}")
