import os
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from backend.core.config import settings

print("--- Initializing Shared Dependencies (LLM, ChromaDB & BM25) ---")

# We need to go up 4 levels to hit the main 'RAG.project copy 2' folder where your model is.
# __file__ is agentic_rag/backend/core/dependencies.py
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CORE_DIR)
AGENTIC_RAG_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(AGENTIC_RAG_DIR) # The main RAG.project copy 2 directory

MODEL_PATH = os.path.join(PROJECT_ROOT, "Phi-3-mini-4k-instruct-q4.gguf")
DB_PATH = os.path.join(PROJECT_ROOT, "db")

# 1. Initialize Embeddings (Loaded once in memory)
print("Loading embeddings...")
embeddings = LlamaCppEmbeddings(
    model_path=MODEL_PATH, 
    n_batch=512, 
    n_ctx=2048
)

# 2. Initialize Vector Store (Loaded once)
print("Loading vector store...")
vector_store = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=embeddings
)

# 3. Initialize BM25 Lexical Index
print("Building BM25 Lexical Index from ChromaDB...")
try:
    # Extract existing documents from ChromaDB to build lexical index
    db_data = vector_store.get()
    bm25_docs = []
    
    # Check if DB has data
    if db_data and 'documents' in db_data and len(db_data['documents']) > 0:
        for i in range(len(db_data['ids'])):
            bm25_docs.append(Document(
                page_content=db_data['documents'][i],
                metadata=db_data['metadatas'][i] if db_data['metadatas'] else {}
            ))
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        print(f"BM25 Index built with {len(bm25_docs)} documents.")
    else:
        print("Warning: ChromaDB is empty. BM25 Index could not be built.")
        bm25_retriever = None
except Exception as e:
    print(f"Error building BM25 index: {e}")
    bm25_retriever = None

# 4. Initialize Shared LLM (Loaded once)
print("Loading LLM...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.7,
    max_tokens=2048,
    top_p=1,
    n_ctx=2048,
    n_gpu_layers=-1,   # Tries to offload to Apple Silicon GPU (Metal)
    n_threads=8,       # Maximize CPU core usage
    n_batch=512,       # Faster prompt processing
    stop=["<|end|>", "<|user|>", "Question:"], # Prevent endless loops
    verbose=False
)

def get_llm():
    return llm

def get_vector_store():
    return vector_store

def get_bm25_retriever():
    return bm25_retriever
