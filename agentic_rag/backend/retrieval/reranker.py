import time
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Initialize globally to avoid reloading the model on every query
# Using the lightweight miniLM model for speed and efficiency
try:
    print("Loading Cross-Encoder Reranker...")
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
except Exception as e:
    print(f"Failed to load Cross-Encoder: {e}")
    reranker_model = None

def rerank_documents(query: str, docs: list[Document], top_k: int = 3) -> list[Document]:
    """
    Reranks a list of fused documents against the original query using a Cross-Encoder.
    Measures and logs reranking latency.
    """
    if not docs or not reranker_model:
        return docs
        
    print(f"\n--- CROSS-ENCODER: Reranking {len(docs)} fused documents ---")
    start_time = time.time()
    
    # Prepare pairs of (query, document_content)
    pairs = [[query, doc.page_content] for doc in docs]
    
    # Predict relevance scores
    scores = reranker_model.predict(pairs)
    
    # Attach scores to documents for sorting
    scored_docs = list(zip(docs, scores))
    
    # Sort by cross-encoder score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Extract top_k
    reranked_docs = [doc for doc, score in scored_docs[:top_k]]
    
    latency = time.time() - start_time
    print(f"  -> Reranking completed in {latency:.2f}s")
    print(f"  -> Top CE Score: {scored_docs[0][1]:.3f} | Lowest CE Score: {scored_docs[-1][1]:.3f}")
    
    return reranked_docs
