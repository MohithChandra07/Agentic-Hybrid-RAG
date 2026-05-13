import time
from backend.core.dependencies import get_vector_store, get_llm
from langchain_core.prompts import PromptTemplate

class BaselineRAG:
    """
    Standard Naive RAG (Vector Only, No Critic, No Retries).
    Used as the control group for the ablation study.
    """
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_llm()
        
    def invoke(self, query: str):
        start_time = time.time()
        
        # 1. Naive Vector Retrieval
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=3)
        else:
            docs = []
            
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. Naive Generation
        template = """<|system|>
You are a helpful assistant. Answer the user's question using the following context.
Context: {context}
<|user|>
Question: {input}
<|assistant|>"""
        prompt = PromptTemplate.from_template(template)
        formatted_prompt = prompt.format(context=context, input=query)
        
        answer = self.llm.invoke(formatted_prompt).strip()
        
        latency = time.time() - start_time
        
        return {
            "retrieved_docs": docs,
            "final_answer": answer,
            "latency": latency,
            "loop_count": 0,
            "relevance_score": 0.0, # Not evaluated in naive RAG
            "retrieval_confidence_score": 0.0
        }
