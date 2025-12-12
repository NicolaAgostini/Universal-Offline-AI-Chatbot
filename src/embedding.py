# === src/embedding.py ===
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from dotenv import load_dotenv
# import os
#
# def get_embedding_model():
#     load_dotenv()  # load from .env
#     hf_token = os.getenv("HF_TOKEN")
#
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"use_auth_token": hf_token}
#     )

from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

