from typing import Annotated, List, TypedDict, Sequence, Dict, Any
import operator
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    Represents the state of our agentic RAG graph.
    As execution progresses through the nodes, this state is updated.
    """
    original_query: str
    search_queries: List[str]
    retrieved_docs: List[Document]
    
    # --- Retrieval Evaluation ---
    retrieval_confidence_score: float
    retrieval_metadata: Dict[str, Any] 
    
    # --- NEW: Retrieval Verification ---
    docs_contradictory: bool
    document_agreement_score: float
    reranking_confidence: float
    
    # --- NEW: Adaptive Policies ---
    query_complexity: float
    retrieval_policy: Dict[str, Any]
    
    # --- NEW: Critic Structured Outputs ---
    relevance_score: float
    hallucination_risk_score: float
    critic_confidence: float # Uncertainty estimation
    retry_recommended: bool
    critic_reasoning: str
    critic_feedback: str
    approved_context: List[Document]
    
    context_sufficient: bool
    draft_answer: str
    hallucination_score: float
    final_answer: str
    loop_count: int
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # --- NEW: Retry Stability ---
    retry_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # --- NEW: Failure Memory ---
    failed_queries: Annotated[List[str], operator.add]
    hallucinated_answers: Annotated[List[str], operator.add]
    
    # --- NEW: Claim Verification ---
    supported_claims: List[str]
    unsupported_claims: List[str]
    answer_confidence: float
