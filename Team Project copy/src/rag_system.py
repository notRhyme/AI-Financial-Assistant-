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
    Lazy loader: Connects to OpenAI only when needed.
    Prevents crash on startup if API key/billing is invalid.
    """
    # 1. Check for API Key first
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY is missing.")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Load existing DB if it exists
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # 3. Create new DB if PDF exists
    if os.path.exists(PDF_PATH):
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
            print(f"❌ RAG INDEXING ERROR: {e}")
            return None
    
    return None

def rag_query(question: str) -> List[Dict]:
    """
    Safe query function that handles errors gracefully.
    """
    try:
        db = get_vectorstore()
        if not db:
            return [{"text": "Database unavailable.", "source": "System"}]

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
        # Catch billing/API errors here so the whole app doesn't die
        return [{"text": f"Error querying database: {e}", "source": "System"}]
