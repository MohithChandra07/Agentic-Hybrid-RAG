import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- IMPORTANT: Update this path ---
MODEL_PATH = "./Phi-3-mini-4k-instruct-q4.gguf" 

# --- 1. Load your dataset ---
DATA_DIR = "./data/"
all_docs = []

print(f"Loading all .txt files from {DATA_DIR}...")

# Loop through all files in the 'data' directory
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(DATA_DIR, filename)
        print(f"Loading {filename}...")
        loader = TextLoader(file_path)
        # Add all documents from this file to our list
        all_docs.extend(loader.load())

if not all_docs:
    print(f"Error: No .txt files found in {DATA_DIR}. Please add your .txt files and try again.")
    exit()

print(f"Loaded a total of {len(all_docs)} documents from all .txt files.")

# --- 2. Split the documents into chunks ---
print("Splitting text...")
# Increased chunk size to 750 for better embedding context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1750, chunk_overlap=50) 
chunks = text_splitter.split_documents(all_docs) 

# --- 3. Initialize the Embedding Model (using your small LLM) ---
print("Initializing embeddings...")
embeddings = LlamaCppEmbeddings(model_path=MODEL_PATH, n_batch=8, n_ctx=2048, verbose=False)

# --- 4. Create the Vector Database ---
print("Creating and saving vector store...")
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="./db"
)

print(f"Success! {len(chunks)} chunks ingested into ChromaDB.")