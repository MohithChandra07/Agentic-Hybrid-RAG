import logging
import json
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from backend.core.state import AgentState

# Define log directory in the project root
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for production structured logging."""
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage()
        }
        if hasattr(record, "rag_telemetry"):
            log_record.update(record.rag_telemetry)
        return json.dumps(log_record)

def setup_rag_logger():
    """Sets up a rotating JSON logger for the Agentic RAG system."""
    logger = logging.getLogger("AgenticRAG")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    log_file = os.path.join(LOGS_DIR, "agentic_rag.log")
    
    # 5MB per file, max 3 backup files
    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    handler.setFormatter(JSONFormatter())
    
    logger.addHandler(handler)
    return logger

rag_logger = setup_rag_logger()

def log_agentic_state(state: AgentState, latency: float):
    """
    Extracts key metrics from the LangGraph AgentState and logs them
    as a structured JSON record for production monitoring.
    """
    approved_docs = state.get("approved_context", [])
    web_fallback_used = any(d.metadata.get("source") == "Internet_DuckDuckGo" for d in approved_docs)
    
    docs_summary = [
        {"source": d.metadata.get("source", "Unknown"), "preview": d.page_content[:100]}
        for d in approved_docs
    ]
    
    telemetry = {
        "original_query": state.get("original_query", ""),
        "rewritten_queries": state.get("search_queries", []),
        "retrieved_docs_count": len(approved_docs),
        "docs_retrieved": docs_summary,
        "critic_relevance_score": state.get("relevance_score", 0.0),
        "critic_confidence": state.get("critic_confidence", 0.0),
        "loop_count": state.get("loop_count", 0),
        "latency_sec": round(latency, 3),
        "hallucination_score": state.get("hallucination_score", 0.0),
        "final_answer_confidence": state.get("claim_confidence_score", 0.0),
        "web_fallback_used": web_fallback_used,
        "failed_queries_count": len(state.get("failed_queries", []))
    }
    
    rag_logger.info("RAG Transaction Complete", extra={"rag_telemetry": telemetry})
