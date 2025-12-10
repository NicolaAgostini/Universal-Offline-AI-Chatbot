# === Requirements ===

import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pyfiglet import figlet_format
from termcolor import colored

# === Load environment (.env should have HF_TOKEN) ===
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")  # Only for embeddings

# === Stylish Heading ===
def stylish_heading():
    title = figlet_format("Lawyer-Bot", font="starwars", width=1000)
    print(colored(title, "green"))

# === Load Local Ollama Model (Offline Mistral) ===
def load_llm():
    llm = ChatOllama(
        model="mistral:instruct",
        temperature=0.5
    )
    return llm

# === Custom Prompt ===
CUSTOM_PROMPT_TEMPLATE = """
Utilizza le informazioni fornite nel contesto per rispondere alla domanda dell'utente.
Se non conosci la risposta, rispondi che non la conosci, non inventarla.
Fai riferimento solo al contesto fornito.

Contesto: {context}
Domanda: {question}

Inizia la risposta direttamente senza testo superfluo.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# === PDF Document Loader ===
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# === Split into Chunks ===
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# === Embedding Model (Requires HF Token) ===
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Save to FAISS DB ===
def build_vector_db(chunks, embedding_model, db_path):
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(db_path)

# === Load from FAISS DB ===
def load_vector_db(db_path, embedding_model):
    return FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

# === Setup QA Chain ===
def setup_qa_chain(llm, db, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# === Paths ===
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# === Pipeline Execution ===
if __name__ == "__main__":
    stylish_heading()
    print("\n📄 Loading documents and building knowledge base...")

    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)
    embedding_model = get_embedding_model()

    if not os.path.exists(DB_FAISS_PATH):
        print("🔧 Creating FAISS database...")
        build_vector_db(text_chunks, embedding_model, DB_FAISS_PATH)

    print("✅ Vector DB ready. Launching QA Chat...\n")

    db = load_vector_db(DB_FAISS_PATH, embedding_model)
    llm = load_llm()
    qa_chain = setup_qa_chain(llm, db, set_custom_prompt(CUSTOM_PROMPT_TEMPLATE))

    print("\n🟢 You can start chatting now. Type 'Exit the Chatbot' to end the session.\n")

    while True:
        user_query = input("🧠 You: ")
        if user_query.strip().lower() == "exit the chatbot":
            print("\n👋 Exiting... Have a great day!\n")
            break

        try:
            print(f"🧠 You: {user_query}")  # Explicitly printing user query
            result = qa_chain.invoke({"query": user_query})
            print(f"🤖 Bot: {result['result']}\n")
        except Exception as e:
            print("❌ Error:", e)
