from backend.core.memory_store import global_memory_store
from backend.core.dependencies import get_llm
from langchain_core.prompts import PromptTemplate

def summarize_and_store_memory(query: str, answer: str, context_docs: list):
    """Summarizes a successful conversation turn and stores it in semantic vector memory."""
    if not answer or answer.startswith("I cannot answer"):
        return
        
    print("\n--- CONVERSATION MEMORY: Synthesizing Turn into Long-Term Memory ---")
    
    # We want a highly condensed, atomic version of what was discussed to save vector space
    template = """<|system|>
You are a Long-Term Memory Extraction AI.
Read the User Query and the System Answer. Condense them into a single, highly factual, objective sentence that captures the core entity and action.
Focus strictly on facts that might be useful for future conversations.
<|user|>
Query: {query}
Answer: {answer}<|end|>
<|assistant|>"""

    llm = get_llm()
    prompt = PromptTemplate.from_template(template)
    
    try:
        summary = llm.invoke(prompt.format(query=query, answer=answer)).strip()
        print(f"  -> Extracted Memory: {summary}")
        
        metadata = {"type": "conversation_turn", "original_query": query}
        global_memory_store.add_memory(summary, metadata)
    except Exception as e:
        print(f"  -> Memory storage failed: {e}")
