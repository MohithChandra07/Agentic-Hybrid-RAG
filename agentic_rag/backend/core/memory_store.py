from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

class LongTermMemoryStore:
    """
    Semantic Vector Store dedicated entirely to Long-Term Conversational Memory.
    """
    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            # Separate persistence directory so it doesn't corrupt the main factual DB
            self.vector_store = Chroma(
                collection_name="conversation_memory", 
                embedding_function=self.embeddings,
                persist_directory="./db_memory"
            )
        except Exception as e:
            print(f"Warning: Memory Store initialization failed. {e}")
            self.vector_store = None
        
    def add_memory(self, content: str, metadata: dict):
        if not self.vector_store: return
        doc = Document(page_content=content, metadata=metadata)
        self.vector_store.add_documents([doc])
        
    def search_memory(self, query: str, k: int = 2) -> list[Document]:
        if not self.vector_store: return []
        try:
            return self.vector_store.similarity_search(query, k=k)
        except:
            return []

# Global Singleton instance
global_memory_store = LongTermMemoryStore()
