from backend.core.state import AgentState
from backend.retrieval.decomposition import decompose_query
from backend.agents.memory_retriever import retrieve_past_memory
from backend.retrieval.query_complexity import assess_complexity
from backend.retrieval.retrieval_policy import apply_retrieval_policy

def planner_node(state: AgentState) -> AgentState:
    """
    Analyzes the original query.
    If this is a retry triggered by the Context Critic, it refines the 
    query using the Critic's explicit feedback.
    """
    query = state.get("original_query", "")
    feedback = state.get("critic_feedback", "")
    loop_count = state.get("loop_count", 0)
    failed_queries = state.get("failed_queries", [])
    
    print(f"\n--- PLANNER NODE (Loop {loop_count}) ---")
    
    decomposed_queries = []
    
    # Adaptive Query Refinement Logic
    if loop_count > 0 and feedback and feedback.upper() != "NONE":
        print(f"  -> Applying Critic Feedback: {feedback}")
        
        # In a full LLM implementation, you would pass `query` + `feedback` 
        # to the LLM here to generate a complex Boolean/Hybrid search string.
        # For efficiency, we append the feedback heuristically to expand recall.
        
        refined_query = f"{query} {feedback}"
        
        # Prevent repeating previously failed actions (Failure Memory)
        if refined_query in failed_queries:
            print("  -> Planner: Warning! This refined query previously failed. Attempting alternative refinement...")
            refined_query = f"{feedback} {query}" # Simple alternative strategy
            if refined_query in failed_queries:
                 refined_query = f"{query} details regarding {feedback}"
                 
        decomposed_queries.append(refined_query)
        print(f"  -> Refined Search Query: '{refined_query}'")
    else:
        # Initial pass: check for multi-hop
        print(f"  -> Initial Search Query: '{query}'")
        
        # CONVERSATIONAL MEMORY: Retrieve past interactions
        past_memory = retrieve_past_memory(query)
        enhanced_query = f"{query} (Prior Context: {past_memory})" if past_memory else query
        
        # ADAPTIVE POLICY: Assess complexity on first pass
        complexity = assess_complexity(enhanced_query)
        policy = apply_retrieval_policy(complexity)
        
        print("  -> Analyzing for Multi-Hop complexity...")
        decomposed_queries = decompose_query(enhanced_query)
        
        if len(decomposed_queries) > 1:
            print(f"  -> Multi-Hop Detected! Decomposed into: {decomposed_queries}")
        else:
            print("  -> Query is simple. Proceeding with standard search.")
            
    return {
        "search_queries": decomposed_queries,
        "query_complexity": complexity if loop_count == 0 else state.get("query_complexity", 0.5),
        "retrieval_policy": policy if loop_count == 0 else state.get("retrieval_policy", {})
    }
