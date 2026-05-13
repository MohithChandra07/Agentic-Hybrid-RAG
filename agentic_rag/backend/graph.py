# pyrefly: ignore [missing-import]
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from backend.core.state import AgentState
from backend.agents.planner import planner_node
from backend.agents.retriever import retriever_node, web_retriever_node
from backend.agents.critic import context_critic_node, answer_critic_node
from backend.agents.generator import generator_node
from backend.agents.verifier import verifier_node
from backend.core.config import settings

def increment_loop(state: AgentState):
    """Utility node to increment the safety loop counter."""
    current_loops = state.get("loop_count", 0)
    print(f"--- SYSTEM: Incrementing loop counter to {current_loops + 1} ---")
    return {"loop_count": current_loops + 1}

def build_graph():
    """
    Constructs the LangGraph state machine with full self-correction routing.
    """
    workflow = StateGraph(AgentState)
    
    # 1. Add Nodes
    workflow.add_node("planner", planner_node) 
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("context_critic", context_critic_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("verifier", verifier_node) # NEW
    workflow.add_node("answer_critic", answer_critic_node)
    workflow.add_node("increment_loop", increment_loop)
    workflow.add_node("web_retriever", web_retriever_node)
    
    # 2. Entry Edge
    workflow.set_entry_point("planner")
    
    # 3. Standard Edges
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "context_critic")
    workflow.add_edge("increment_loop", "planner")
    workflow.add_edge("web_retriever", "generator") # WEB feeds directly to generator
    
    # 4. Adaptive Routing: The Context Critic
    def decide_to_generate(state: AgentState):
        """Routes from Context Critic based on relevance, confidence, loop safety, and diminishing returns."""
        is_sufficient = state.get("context_sufficient", False)
        loop_count = state.get("loop_count", 0)
        critic_confidence = state.get("critic_confidence", 1.0)
        retry_history = state.get("retry_history", [])
        
        if is_sufficient:
            print("  -> Routing: Context Approved. Proceeding to Generation.")
            return "generator"
            
        # 1. Uncertainty Guard: Prevent blind loops if Critic is just guessing
        if critic_confidence < 0.5:
            print(f"  -> Routing: Critic Confidence too low ({critic_confidence}). Skipping internal retry loop and escalating to WEB SEARCH.")
            return "web_retriever"
            
        # 2. Adaptive Stopping Criteria: Detect Diminishing Returns
        if loop_count > 0 and len(retry_history) >= 2:
            prev_score = retry_history[-2]["relevance_score"]
            curr_score = retry_history[-1]["relevance_score"]
            improvement = curr_score - prev_score
            
            print(f"  -> Retry Effectiveness: {improvement:+.2f} relevance score change")
            
            # If the score went down, or didn't improve by at least 0.05, we are stuck in a low-value retry loop.
            if improvement <= 0.05:
                print("  -> Routing: Diminishing returns detected. Retries are no longer effective. Escalating to WEB SEARCH.")
                return "web_retriever"
                
        # 3. Hard Safety Limit
        policy = state.get("retrieval_policy", {})
        dynamic_max_retries = policy.get("max_retries", settings.MAX_RETRIES)
        
        if loop_count >= dynamic_max_retries:
            print(f"  -> Routing: Adaptive safety limit reached ({dynamic_max_retries} retries). Escalating to WEB SEARCH FALLBACK.")
            return "web_retriever" # Jump to internet if internal DB fails
            
        print("  -> Routing: Context Insufficient but improving. Triggering Retrieval Retry.")
        return "increment_loop"
        
    workflow.add_conditional_edges(
        "context_critic",
        decide_to_generate,
        {
            "generator": "generator",
            "increment_loop": "increment_loop",
            "web_retriever": "web_retriever"
        }
    )
    
    # 5. Verification & Hallucination Routing
    workflow.add_edge("generator", "verifier")
    workflow.add_edge("verifier", "answer_critic")
    
    def decide_final_state(state: AgentState):
        """Routes from Answer Critic. Refuses hallucinated answers."""
        score = state.get("hallucination_score", 0.0)
        loop_count = state.get("loop_count", 0)
        policy = state.get("retrieval_policy", {})
        dynamic_max_retries = policy.get("max_retries", settings.MAX_RETRIES)
        
        if score > 0.5:
            if loop_count >= dynamic_max_retries:
                print(f"  -> Routing: Hallucination detected, but adaptive max retries ({dynamic_max_retries}) reached. Halting.")
                return END
            print("  -> Routing: Hallucination detected! Forcing regeneration.")
            # We route back to increment_loop to completely restart the process
            return "increment_loop" 
            
        print("  -> Routing: Answer verified. Finishing execution.")
        return END
        
    workflow.add_conditional_edges(
        "answer_critic",
        decide_final_state,
        {
            "increment_loop": "increment_loop",
            END: END
        }
    )
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_graph()
    
    print("\n=======================================================")
    print(" CRITIC-GUIDED AGENTIC RAG INITIALIZED ")
    print("=======================================================\n")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("\n> ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue
            
        initial_state = {
            "original_query": query,
            "loop_count": 0
        }
        
        # Stream the graph execution and print nodes as they finish
        for output in app.stream(initial_state, config={"recursion_limit": 100}):
            for key, value in output.items():
                pass # The nodes themselves have print statements explaining what they do
                
        # To get the final answer, we can invoke the graph instead of stream, 
        # or just let the generator node print it.
        # But for clean output, let's fetch the final state
        final_state = app.invoke(initial_state, config={"recursion_limit": 100})
        print("\n================ FINAL ANSWER ================")
        print(final_state.get("final_answer", "No answer generated."))
        print("==============================================\n")
