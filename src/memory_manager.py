import os
from langchain_community.vectorstores import FAISS
from src.config import MEMORY_DB_PATH



def load_memory_db(embedding_model):
    """Load or create persistent FAISS memory DB"""

    index_file = os.path.join(MEMORY_DB_PATH, "index.faiss")
    store_file = os.path.join(MEMORY_DB_PATH, "index.pkl")

    # if db doesn't exist
    if not os.path.exists(index_file) or not os.path.exists(store_file):
        print("🧠 Memory DB not found → creating new one")
        os.makedirs(MEMORY_DB_PATH, exist_ok=True)

        memory = FAISS.from_texts(
            ["initial memory"],  # serve almeno 1 documento
            embedding_model
        )
        memory.save_local(MEMORY_DB_PATH)
        return memory

    #if db esists
    try:
        memory = FAISS.load_local(
            MEMORY_DB_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print("🧠 Memory DB Loaded.")
        return memory

    except Exception as e:
        print("❌ Memory DB corrupted → recreating:", e)

        memory = FAISS.from_texts(
            ["initial memory"],
            embedding_model
        )
        memory.save_local(MEMORY_DB_PATH)
        return memory


def save_memory(text, memory_db):
    """Save memory DB"""
    memory_db.add_texts([text])
    memory_db.save_local(MEMORY_DB_PATH)
    print("💾 Memory saved:", text)





