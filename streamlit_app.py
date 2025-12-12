# streamlit_app.py



import streamlit as st
from sympy.codegen import Print

from src.config import DATA_PATH, DB_FAISS_PATH, MISC_PATH, MODEL
from src.utils import stylish_heading
from src.loader import load_pdf_files
from src.chunker import create_chunks
from src.embedding import get_embedding_model
from src.vectorstore import build_vector_db, load_vector_db
from src.model_loader import load_llm
from src.prompts import CUSTOM_PROMPT_TEMPLATE, set_custom_prompt
from src.qa_chain import setup_qa_chain
from src.memory_manager import load_memory_db, save_memory
from src.translator import translate_to_italian, translate_to_english
import os

# Page config
st.set_page_config(page_title="J.A.R.V.I.S.", page_icon="🧠", layout="centered")

# Header
stylish_heading()

st.markdown("✅ Powered by **Offline LLama** + **FAISS**")
st.image(os.path.join(MISC_PATH, "logo.png"))

st.divider()

# Load pipeline once
@st.cache_resource(show_spinner="Warming up the brain... 🧠⚙️")
def load_pipeline():
    documents = load_pdf_files(DATA_PATH)
    chunks = create_chunks(documents)
    embedding_model = get_embedding_model()

    if not os.path.exists(DB_FAISS_PATH):
        build_vector_db(chunks, embedding_model, DB_FAISS_PATH)

    db_main = load_vector_db(DB_FAISS_PATH, embedding_model)
    db_memory = load_memory_db(embedding_model)
    llm = load_llm()
    qa_chain = setup_qa_chain(llm, db_main, db_memory, set_custom_prompt())
    return qa_chain, db_memory, embedding_model

qa_chain, db_memory, embedding_model = load_pipeline()

# Start chat session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input/output loop
st.chat_message("assistant").markdown("👋 Hello! I'm your AI assistant. Ask me anything about your uploaded documents.")

user_input = st.chat_input("Type your question here...")

if user_input:

    # --- Learning mode ---
    if user_input.lower().startswith("learn:"):
        new_knowledge_en = translate_to_english(user_input.replace("learn:", "").strip()) #translate to english
        save_memory(new_knowledge_en, db_memory)
        st.chat_message("assistant").markdown(f"🧠 I learned: **{new_knowledge_en}**")
    else:
        # Normal question
        st.chat_message("user").markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🧠"):
                # try:
                query_en = translate_to_english(user_input) #translate to english

                print("Question in english:", query_en)

                answer_en = qa_chain.invoke({"query": query_en})["result"]
                print("Answer in english:", answer_en)

                answer_it = translate_to_italian(answer_en)# translate to italian

                st.markdown(f"🤖 {answer_it}")
                st.session_state.chat_history.append(("bot", answer_en)) #append answer in english for chat session memory
                # except Exception as e:
                #     st.error(f"❌ Error: {e}")



# Optionally: show previous messages
if st.session_state.chat_history:
    st.divider()
    st.markdown("🕓 **Conversation History**", help="Scroll back through your past questions and answers.")
    for sender, msg in st.session_state.chat_history:
        icon = "🧠" if sender == "user" else "🤖"
        st.markdown(f"**{icon} {sender.capitalize()}**: {msg}")
