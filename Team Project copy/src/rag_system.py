import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# === CONFIG ===
PDF_PATH = "data/finance.pdf"
PERSIST_DIR = "chroma_finance_db"

def get_vectorstore():
    """
    Lazy loader for the vector store. 
    Only connects to OpenAI when this function is actually called.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    if not os.path.exists(PDF_PATH):
        # Fallback for testing if PDF is missing
        print(f"⚠️ WARNING: {PDF_PATH} not found. RAG will not work.")
        return None

    print("--- ⚙️ Indexing PDF (This costs API credits) ---")
    try:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        return db
    except Exception as e:
        print(f"❌ RAG SETUP ERROR: {e}")
        return None

def rag_query(question: str) -> List[Dict]:
    """
    Safe query function that handles missing DB or API errors gracefully.
    """
    try:
        db = get_vectorstore() # Initialize only when needed
        if not db:
            return [{"text": "Database not initialized.", "source": "System"}]

        docs = db.similarity_search(question, k=3)
        results = []
        for d in docs:
            meta = d.metadata or {}
            results.append({
                "text": d.page_content,
                "source": meta.get("source", "doc"),
                "page": meta.get("page", 0)
            })
        return results
    except Exception as e:
        return [{"text": f"Error querying database: {e}", "source": "System"}]