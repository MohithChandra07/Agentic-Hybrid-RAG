from backend.core.memory_store import global_memory_store

def retrieve_past_memory(query: str) -> str:
    """Retrieves relevant past conversation context from vector memory."""
    docs = global_memory_store.search_memory(query, k=2)
    if not docs:
        return ""
        
    print(f"\n--- MEMORY RETRIEVER: Activated ---")
    print(f"  -> Retrieved {len(docs)} relevant past semantic memories.")
    
    memories = []
    for doc in docs:
        memories.append(doc.page_content)
        print(f"  -> Memory context: {doc.page_content}")
        
    return "\n".join(memories)
