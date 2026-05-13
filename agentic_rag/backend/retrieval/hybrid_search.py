from typing import List, Dict, Tuple
from langchain_core.documents import Document
from backend.core.dependencies import get_vector_store, get_bm25_retriever

class HybridRetriever:
    """
    True production-grade Hybrid Retrieval pipeline.
    Combines Vector Similarity (Chroma) and Lexical Keyword Match (BM25)
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, k_vector: int = 5, k_bm25: int = 5, rrf_k: int = 60, final_k: int = 3):
        self.vector_store = get_vector_store()
        self.bm25_retriever = get_bm25_retriever()
        
        self.k_vector = k_vector  # How many to pull from Vector DB
        self.k_bm25 = k_bm25      # How many to pull from BM25
        self.rrf_k = rrf_k        # RRF smoothing constant (usually 60)
        self.final_k = final_k    # Final number of fused documents to return

    def retrieve(self, queries: List[str]) -> Tuple[List[Document], float, Dict]:
        """
        Executes hybrid search across the vector database and lexical index.
        Returns: (Fused Documents, Confidence Score, Metadata)
        """
        all_vector_docs = []
        all_bm25_docs = []
        
        # 1. Execute Searches
        for query in queries:
            print(f"--- HYBRID PIPELINE: Searching for '{query}' ---")
            
            # Vector Search (Dense)
            if self.vector_store:
                v_docs = self.vector_store.similarity_search_with_relevance_scores(query, k=self.k_vector)
                all_vector_docs.extend(v_docs)
                print(f"  -> Vector Search returned {len(v_docs)} hits")
                
            # BM25 Search (Sparse/Lexical)
            if self.bm25_retriever:
                # BM25Retriever doesn't return scores natively in all versions, 
                # so we just get docs and rely on rank order for RRF
                self.bm25_retriever.k = self.k_bm25
                b_docs = self.bm25_retriever.invoke(query)
                all_bm25_docs.extend(b_docs)
                print(f"  -> BM25 Lexical Search returned {len(b_docs)} hits")

        # 2. Reciprocal Rank Fusion (RRF)
        fused_results, max_score = self._apply_rrf(all_vector_docs, all_bm25_docs)
        
        # 3. Deduplicate and Extract Final Docs
        final_docs = []
        seen_content = set()
        
        for doc, score in fused_results:
            if doc.page_content not in seen_content:
                # Inject the RRF score into the document metadata for transparency
                doc.metadata["rrf_score"] = score
                seen_content.add(doc.page_content)
                final_docs.append(doc)
            if len(final_docs) >= self.final_k:
                break
                
        # Calculate a pseudo-confidence score (0.0 to 1.0 heuristic based on max RRF score)
        # Max theoretical single-query score in RRF is (1/61) + (1/61) = 0.0327
        # We normalize this for logging readability
        confidence = min(max_score / 0.0327, 1.0) if max_score > 0 else 0.0
        
        metadata = {
            "total_vector_hits": len(all_vector_docs),
            "total_bm25_hits": len(all_bm25_docs),
            "fused_hits": len(fused_results)
        }
        
        return final_docs, confidence, metadata

    def _apply_rrf(self, vector_hits: List[Tuple[Document, float]], bm25_hits: List[Document]) -> Tuple[List[Tuple[Document, float]], float]:
        """
        Applies Reciprocal Rank Fusion to merge and rerank result lists.
        Formula: score = 1 / (rank + k)
        """
        doc_scores = {}
        doc_map = {}
        
        # Process Vector Hits (List of Tuples: (Doc, Score))
        for rank, (doc, _) in enumerate(vector_hits):
            content_hash = hash(doc.page_content)
            if content_hash not in doc_scores:
                doc_scores[content_hash] = 0.0
                doc_map[content_hash] = doc
            doc_scores[content_hash] += 1.0 / (rank + 1 + self.rrf_k)
            
        # Process BM25 Hits (List of Docs)
        for rank, doc in enumerate(bm25_hits):
            content_hash = hash(doc.page_content)
            if content_hash not in doc_scores:
                doc_scores[content_hash] = 0.0
                doc_map[content_hash] = doc
            doc_scores[content_hash] += 1.0 / (rank + 1 + self.rrf_k)
            
        # Sort documents by their combined RRF score in descending order
        sorted_hashes = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        fused_docs = [(doc_map[h], doc_scores[h]) for h in sorted_hashes]
        max_score = doc_scores[sorted_hashes[0]] if sorted_hashes else 0.0
        
        return fused_docs, max_score
