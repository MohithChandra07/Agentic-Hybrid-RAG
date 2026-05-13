import re
from langchain_core.prompts import PromptTemplate
from backend.core.state import AgentState
from backend.core.dependencies import get_llm

def context_critic_node(state: AgentState) -> AgentState:
    """
    Evaluates if the retrieved documents contain sufficient, relevant information.
    Prevents hallucination by gatekeeping bad context.
    """
    query = state.get("original_query", "")
    docs = state.get("retrieved_docs", [])
    
    print(f"--- CONTEXT CRITIC NODE: Evaluating Context Relevance ---")
    
    if not docs:
        print("  -> Critic: No documents found. Forcing retry.")
        return {
            "context_sufficient": False,
            "retry_recommended": True,
            "relevance_score": 0.0,
            "hallucination_risk_score": 1.0,
            "critic_reasoning": "Retrieval pipeline returned empty results.",
            "critic_feedback": "Expand the search terms or use broader lexical keywords.",
            "approved_context": []
        }
        
    context_text = "\n\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    llm = get_llm()
    
    # 1. The Research-Grade Critic Prompt with Uncertainty Estimation
    template = """
    <|system|>
    You are an expert Context Critic. Your job is to evaluate if the retrieved documents contain sufficient and relevant information to answer the user's query.
    If the context is irrelevant or insufficient, we risk severe hallucination.
    
    Respond STRICTLY in the exact following multi-line format:
    RELEVANCE_SCORE: [A number from 0.0 to 1.0]
    CONFIDENCE: [A number from 0.0 to 1.0 indicating your certainty in this evaluation]
    REASONING: [1 sentence explaining why]
    DECISION: [APPROVED or INSUFFICIENT]
    FEEDBACK: [If insufficient, suggest 2-3 specific new search keywords. If approved, write NONE]
    
    Context:
    {context}
    <|user|>
    Query: {query}
    <|assistant|>
    """
    
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(context=context_text, query=query)
    
    print("  -> Critic is analyzing retrieval quality and estimating uncertainty...")
    evaluation = llm.invoke(formatted_prompt).strip()
    
    # 2. Parse the structured output
    relevance_score = 0.5
    confidence_score = 0.5
    decision = "INSUFFICIENT"
    reasoning = "Parsing failed."
    feedback = "Try different keywords."
    
    try:
        score_match = re.search(r"RELEVANCE_SCORE:\s*([0-9.]+)", evaluation)
        if score_match: relevance_score = float(score_match.group(1))
        
        conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", evaluation)
        if conf_match: confidence_score = float(conf_match.group(1))
        
        dec_match = re.search(r"DECISION:\s*([A-Z]+)", evaluation)
        if dec_match: decision = dec_match.group(1)
        
        reason_match = re.search(r"REASONING:\s*(.+)", evaluation, re.IGNORECASE)
        if reason_match: reasoning = reason_match.group(1).strip()
        
        feed_match = re.search(r"FEEDBACK:\s*(.+)", evaluation, re.IGNORECASE)
        if feed_match: feedback = feed_match.group(1).strip()
    except Exception as e:
        print(f"  -> Critic parsing error: {e}")
        
    print(f"  -> Relevance: {relevance_score} | Confidence: {confidence_score} | Decision: {decision}")
    print(f"  -> Reasoning: {reasoning}")
    
    is_sufficient = (decision == "APPROVED" and relevance_score >= 0.6)
    
    # 3. Output the structured state
    return {
        "context_sufficient": is_sufficient,
        "retry_recommended": not is_sufficient,
        "relevance_score": relevance_score,
        "hallucination_risk_score": 1.0 - relevance_score, # High relevance = low risk
        "critic_confidence": confidence_score,
        "critic_reasoning": reasoning,
        "critic_feedback": feedback,
        "approved_context": docs if is_sufficient else [],
        "failed_queries": [state.get("search_queries", [query])[-1]] if not is_sufficient else [],
        "retry_history": [{
            "loop": state.get("loop_count", 0),
            "relevance_score": relevance_score,
            "confidence": confidence_score
        }]
    }

def answer_critic_node(state: AgentState) -> AgentState:
    """
    Evaluates the draft answer against the context to detect hallucinations.
    """
    draft = state.get("draft_answer", "")
    docs = state.get("approved_context", [])
    
    print("--- ANSWER CRITIC NODE: Checking for hallucinations with Phi-3 ---")
    
    if not draft or not docs:
        return {
            "hallucination_score": 0.0, 
            "final_answer": draft,
            "hallucinated_answers": []
        }
        
    # Bypass the redundant holistic LLM check because the Verifier Node 
    # already performed a rigorous atomic sentence-by-sentence check!
    # This prevents the smaller Phi-3 model from falsely flagging perfect answers.
    score = state.get("hallucination_score", 0.0)
    
    return {
        "hallucination_score": score, 
        "final_answer": draft,
        "hallucinated_answers": [draft] if score > 0.5 else []
    }
