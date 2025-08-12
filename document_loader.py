# document_loader.py
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_publications(data_dir="."):  # ← Default to current directory
    docs = []
    supported_ext = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".html": UnstructuredHTMLLoader,
        ".txt": TextLoader,
        ".json": TextLoader  # basic support
    }

    print(f"📂 Scanning directory: {os.path.abspath(data_dir)}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in supported_ext:
            try:
                loader = supported_ext[ext](filepath)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = filename  # track original file
                docs.extend(loaded_docs)
                print(f"✅ Loaded {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
        else:
            print(f"⚠️ Skipped unsupported file: {filename}")

    # Split docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    print(f"🧩 Created {len(split_docs)} text chunks.")
    return split_docs