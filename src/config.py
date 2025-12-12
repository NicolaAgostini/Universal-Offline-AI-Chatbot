# === src/config.py ===
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_PATH = "data/"
MISC_PATH = "misc/"
DB_FAISS_PATH = "vectorstore/db_faiss"

MEMORY_DB_PATH = "memory_db"

MODEL = "llama3.1:8b"