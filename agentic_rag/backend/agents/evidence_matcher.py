import re
from langchain_core.prompts import PromptTemplate

def match_evidence(claim: str, context: str, llm) -> float:
    """Matches a single atomic claim against the retrieved context to compute support score."""
    template = """<|system|>
Evaluate if the following claim is strictly supported by the provided context.
Output SUPPORT_SCORE: 1.0 if fully supported, 0.5 if partially supported, 0.0 if not supported.
Context:
{context}
<|user|>
Claim: {claim}<|end|>
<|assistant|>"""
    
    prompt = PromptTemplate.from_template(template)
    response = llm.invoke(prompt.format(context=context, claim=claim)).strip()
    
    match = re.search(r"SUPPORT_SCORE:\s*([0-9.]+)", response, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            return 0.0
    return 0.0
