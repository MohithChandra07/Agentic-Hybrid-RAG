from backend.core.state import AgentState
from backend.core.dependencies import get_llm
from backend.agents.claim_extractor import extract_claims
from backend.agents.evidence_matcher import match_evidence
from backend.agents.conversation_memory import summarize_and_store_memory

def verifier_node(state: AgentState) -> AgentState:
    """
    Lightweight claim verification layer AFTER answer generation.
    Splits the draft into atomic claims, matches them against the context,
    and calculates an overall claim confidence and hallucination score.
    """
    print("\n--- VERIFIER NODE: Running atomic claim verification ---")
    draft = state.get("draft_answer", "")
    docs = state.get("approved_context", [])
    
    if not draft or not docs or draft.startswith("I cannot answer"):
        print("  -> Verifier: Skipping (No valid draft or context)")
        return {
            "supported_claims": [], 
            "unsupported_claims": [], 
            "answer_confidence": 1.0,
            "hallucination_score": 0.0
        }
         
    llm = get_llm()
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    print("  -> Extracting atomic claims from draft...")
    claims = extract_claims(draft, llm)
    supported = []
    unsupported = []
    total_score = 0.0
    
    if not claims:
        print("  -> No atomic claims extracted.")
        return {
            "supported_claims": [], 
            "unsupported_claims": [], 
            "answer_confidence": 1.0,
            "hallucination_score": 0.0
        }
    
    print(f"  -> Matching {len(claims)} claims against evidence...")
    for claim in claims:
        score = match_evidence(claim, context_text, llm)
        total_score += score
        if score >= 0.5:
            supported.append(claim)
        else:
            unsupported.append(claim)
            
    confidence = total_score / len(claims)
    hallucination_score = 1.0 - confidence
    
    print(f"  -> Supported: {len(supported)} | Unsupported (Hallucination Risks): {len(unsupported)}")
    print(f"  -> Answer Confidence: {confidence:.2f} | Hallucination Score: {hallucination_score:.2f}")
    
    # NEW: Store successful answers into Long-Term Semantic Memory
    if hallucination_score < 0.5:
        summarize_and_store_memory(state.get("original_query", ""), draft, docs)
        
    return {
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "answer_confidence": confidence,
        "hallucination_score": hallucination_score
    }
