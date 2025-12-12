import os
from langchain_community.vectorstores import FAISS
from src.config import MEMORY_DB_PATH


def load_memory_db(embedding_model):
    """Load or create memory database"""
    if not os.path.exists(MEMORY_DB_PATH):
        os.makedirs(MEMORY_DB_PATH, exist_ok=True)
        print("🧠 Creating memory DB...")
        return FAISS.from_texts([""], embedding_model)  # DB vuoto
    try:
        memory = FAISS.load_local(MEMORY_DB_PATH,
                            embeddings=embedding_model,
                            allow_dangerous_deserialization=True)
        print("🧠 Memory DB Loaded.")
        return memory

    except RuntimeError as e:
        print("⚠️ FAISS DB not found, creating new one:", e)
        return FAISS.from_texts([""], embedding_model)

    except Exception as e:
        print("❌ Unexpected error while loading FAISS memory DB:", e)


def save_memory(text, memory_db):
    """Save memory DB"""
    memory_db.add_texts([text])
    memory_db.save_local(MEMORY_DB_PATH)
    print("💾 Memory saved:", text)
