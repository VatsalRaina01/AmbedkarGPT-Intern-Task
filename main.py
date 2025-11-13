#!/usr/bin/env python3
"""
AmbedkarGPT 🪶
Simple Command-Line Q&A System built for Kalpit Pvt Ltd – AI Intern Task (Phase 1)

Author: Vatsal Raina
About: A small local Retrieval-Augmented Generation (RAG) prototype that answers questions
       based on Dr. B.R. Ambedkar’s excerpt from *Annihilation of Caste*.
Stack:
- LangChain for pipeline orchestration
- ChromaDB for local vector storage
- HuggingFaceEmbeddings (all-MiniLM-L6-v2) for embeddings
- Ollama + Mistral-7B for local LLM inference
Everything runs 100% offline 💻✨
"""


import sys
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# ---------- CONFIG ----------
SPEECH_FILE = Path("speech.txt")
DB_DIR = Path(".chroma_store")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "phi"

def prepare_vectorstore():
    """Load text, split it into chunks, embed them, and store locally in ChromaDB."""
    if not SPEECH_FILE.exists():
        print("❌ speech.txt missing! Please add the file first.")
        sys.exit(1)

    # Load and split the document
    loader = TextLoader(str(SPEECH_FILE), encoding="utf-8")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunks = splitter.split_documents(docs)

    # Embed + store
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    print("🧩 Creating local vector store...")
    vectordb = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=str(DB_DIR)
    )
    vectordb.persist()
    print("✅ Vector store ready and saved!")
    return vectordb

def load_vectorstore():
    """Reload stored embeddings if already created."""
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

def build_chain():
    """Create the RetrievalQA chain using LangChain."""
    vectordb = load_vectorstore() if DB_DIR.exists() else prepare_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model=LLM_MODEL, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True
    )
    return qa_chain

def ask_question(chain):
    """Interactive Q&A loop."""
    print("\n💬 AmbedkarGPT is live! (type 'exit' to quit)")
    while True:
        query = input("\n🧠 Ask your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("\n👋 Exiting... see you next experiment!")
            break
        if not query:
            continue

        result = chain.invoke({"query": query})
        print(f"\n🗣️ Answer:\n{result['result']}")
        print("\n📚 Context chunks used:")
        for i, doc in enumerate(result["source_documents"], start=1):
            print(f"  {i}. {doc.page_content[:80]}...")

if __name__ == "__main__":
    print("🚀 Booting AmbedkarGPT Prototype...\n")
    qa_chain = build_chain()
    ask_question(qa_chain)
