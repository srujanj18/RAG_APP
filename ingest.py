# ingest.py
from document_loader import load_publications
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def ingest_documents():
    print("📥 Loading and processing Ready Tensor publications...")
    docs = load_publications()

    print("🧠 Generating embeddings (using all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📦 Building FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_ready_tensor")

    print("✅ Ingestion complete! Vector store saved to 'faiss_ready_tensor/'")

if __name__ == "__main__":
    ingest_documents()