def apply_retrieval_policy(complexity_score: float) -> dict:
    """
    Determines retrieval depth, reranking usage, and retry budgets 
    based on the dynamic query complexity score.
    """
    print(f"\n--- ADAPTIVE POLICY: Applying retrieval strategy for complexity {complexity_score:.2f} ---")
    
    if complexity_score < 0.4:
        # Simple factual query -> Fast, shallow retrieval, no heavy reranking
        policy = {
            "strategy": "Shallow (Fast)",
            "k_vector": 3,
            "k_bm25": 3,
            "final_k": 3,
            "max_retries": 1,
            "use_cross_encoder": False
        }
    elif complexity_score < 0.8:
        # Moderate query -> Standard retrieval depth, cross-encoder enabled
        policy = {
            "strategy": "Standard",
            "k_vector": 8,
            "k_bm25": 8,
            "final_k": 10,
            "max_retries": 3,
            "use_cross_encoder": True
        }
    else:
        # Complex multi-hop query -> Deep retrieval, maximum retries
        policy = {
            "strategy": "Deep Analytical",
            "k_vector": 15,
            "k_bm25": 15,
            "final_k": 15,
            "max_retries": 5,
            "use_cross_encoder": True
        }
        
    # OVERRIDE: Forced Fast Strategy for High-Speed Demonstration
    fast_policy = {
        "strategy": "Shallow (Fast)",
        "k_vector": 3,
        "k_bm25": 3,
        "final_k": 3,
        "max_retries": 1,
        "use_cross_encoder": False
    }
    
    print(f"  -> Strategy Selected: {fast_policy['strategy']} (SPEED OVERRIDE ENABLED)")
    print(f"  -> Vector Pull: {fast_policy['k_vector']} | BM25 Pull: {fast_policy['k_bm25']} | Retry Budget: {fast_policy['max_retries']}")
    print(f"  -> Heavy Reranking Enabled: {fast_policy['use_cross_encoder']}")
    
    return fast_policy
