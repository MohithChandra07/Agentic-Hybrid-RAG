from langchain_core.prompts import PromptTemplate

def extract_claims(answer: str, llm) -> list[str]:
    """Splits a generated answer into atomic factual claims."""
    template = """<|system|>
You are an expert at extracting atomic factual claims from text. 
Break down the provided text into independent, atomic factual statements.
Output each claim on a new line starting with "- ".
<|user|>
Text: {text}<|end|>
<|assistant|>"""
    
    prompt = PromptTemplate.from_template(template)
    response = llm.invoke(prompt.format(text=answer)).strip()
    
    claims = []
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('- '):
            claims.append(line[2:].strip())
            
    return claims
