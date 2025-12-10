# === src/loader.py ===
# === src/loader.py ===

import os
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredEPubLoader,
    CSVLoader,
)
from docx import Document as DocxDocument
import pandas as pd
from odf import text, teletype
from odf.opendocument import load as load_odt


SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".csv",
    ".docx",
    ".epub",
    ".odt",
]


def load_docx_file(path: str):
    """Load docx file and read header footer
    """
    try:
        doc = DocxDocument(path)
    except Exception as e:
        print(f"❌ Error opening DOCX {path}: {e}")
        return []

    parts = []

    # paragrafi
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())

    # tabelle (celle)
    try:
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text = cell.text.strip()
                    if text:
                        parts.append(text)
    except Exception as e:
        # non bloccante: continua anche se ci sono problemi con tabelle
        print(f"⚠️ Error reading tables in {path}: {e}")

    # header/footer (se presenti)
    try:
        if doc.sections:
            for sec in doc.sections:
                hdr = sec.header
                ftr = sec.footer
                if hdr and hdr.paragraphs:
                    for p in hdr.paragraphs:
                        if p.text and p.text.strip():
                            parts.append(p.text.strip())
                if ftr and ftr.paragraphs:
                    for p in ftr.paragraphs:
                        if p.text and p.text.strip():
                            parts.append(p.text.strip())
    except Exception:
        # ignoriamo eventuali problemi nella sezione
        pass

    full_text = "\n\n".join(parts).strip()
    if not full_text:
        print(f"⚠️ DOCX {path} loaded but empty.")
        return []

    return [Document(page_content=full_text, metadata={"source": path})]



def load_csv_file(path: str):
    """Carica CSV come testo concatenato."""
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    return [Document(page_content=df.to_string(), metadata={"source": path})]


def load_single_file(path: str):
    """Restituisce un documento LangChain in base al tipo file."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(path).load()

    elif ext == ".txt":
        return TextLoader(path, encoding="utf-8").load()

    elif ext == ".md":
        return UnstructuredMarkdownLoader(path).load()

    elif ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader(path).load()

    elif ext == ".epub":
        return UnstructuredEPubLoader(path).load()

    elif ext == ".csv":
        return load_csv_file(path)

    elif ext == ".docx":
        return load_docx_file(path)

    elif ext == ".odt":
        return load_odt_file(path)

    else:
        print(f"❌ Formato non supportato: {ext} → ignorato.")
        return []


def load_pdf_files(data_path):
    """
    Manteniamo il nome originale della funzione,
    ma ora carica TUTTI i formati supportati.
    """
    documents = []

    for filename in os.listdir(data_path):
        path = os.path.join(data_path, filename)

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(filename)[1].lower()

        if ext in SUPPORTED_EXTENSIONS:
            print(f"📄 Caricamento file: {filename}")
            docs = load_single_file(path)
            documents.extend(docs)
        else:
            print(f"⚠️ Ignorato (formato non supportato): {filename}")

    print(f"📚 Totale documenti caricati: {len(documents)}")
    return documents


def load_odt_file(path: str):
    """Carica file ODT in un singolo documento."""
    doc = load_odt(path)
    all_text = []
    for elem in doc.getElementsByType(text.P):
        all_text.append(teletype.extractText(elem))
    return [Document(page_content="\n".join(all_text), metadata={"source": path})]


