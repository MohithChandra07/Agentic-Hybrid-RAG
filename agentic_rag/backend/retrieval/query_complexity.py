import re
from langchain_core.prompts import PromptTemplate
from backend.core.dependencies import get_llm

def assess_complexity(query: str) -> float:
    """Scores query complexity from 0.0 (simple fact) to 1.0 (complex multi-hop)."""
    llm = get_llm()
    template = """<|system|>
You are an advanced Query Complexity Analyzer.
Read the user's query and calculate a difficulty score.
- Score 0.0 to 0.3 for simple, single-entity factual lookups (e.g., "What is the capital of France?").
- Score 0.4 to 0.7 for moderate queries requiring some reasoning.
- Score 0.8 to 1.0 for highly complex, multi-hop, analytical, or comparative queries.

Output EXACTLY in this format:
COMPLEXITY_SCORE: [score]
<|user|>
Query: {query}<|end|>
<|assistant|>"""
    
    prompt = PromptTemplate.from_template(template)
    try:
        response = llm.invoke(prompt.format(query=query)).strip()
        match = re.search(r"COMPLEXITY_SCORE:\s*([0-9.]+)", response, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"  -> Complexity Assessment Failed: {e}")
        
    return 0.5 # Default to moderate complexity if parsing fails
