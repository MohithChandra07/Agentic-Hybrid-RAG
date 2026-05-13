from backend.core.dependencies import get_llm
from langchain_core.prompts import PromptTemplate
import re

def decompose_query(query: str) -> list[str]:
    """Detects multi-hop queries and breaks them down into sub-queries."""
    llm = get_llm()
    
    template = """<|system|>
You are an advanced Query Decomposition Agent.
Analyze the user's query. If the query requires multiple steps of reasoning, asks to compare two distinct concepts, or is highly complex (Multi-Hop), break it down into 2-3 simpler sub-queries.
If the query is a simple single-step question, just output the original query.

Output EXACTLY in this format:
SUBQUERY: [query 1]
SUBQUERY: [query 2]
...
<|user|>
Query: {query}<|end|>
<|assistant|>"""

    prompt = PromptTemplate.from_template(template)
    response = llm.invoke(prompt.format(query=query)).strip()
    
    subqueries = []
    for line in response.split('\n'):
        match = re.search(r"SUBQUERY:\s*(.*)", line, re.IGNORECASE)
        if match:
            sq = match.group(1).strip()
            if sq:
                subqueries.append(sq)
                
    # Fallback if the LLM fails to format correctly
    if not subqueries:
        return [query]
        
    return subqueries
