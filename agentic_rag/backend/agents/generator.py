from langchain_core.prompts import PromptTemplate
from backend.core.state import AgentState
from backend.core.dependencies import get_llm

def generator_node(state: AgentState) -> AgentState:
    """
    Generates the draft answer STRICTLY using only the approved context.
    If context is empty, it refuses to answer.
    """
    docs = state.get("approved_context", [])
    query = state.get("original_query", "")
    failed_drafts = state.get("hallucinated_answers", [])
    
    print("--- GENERATOR NODE: Drafting final answer ---")
    
    # 1. Hard Fallback: Prevent Hallucination at the source
    if not docs:
        print("  -> Generator: No approved context available. Refusing to answer.")
        return {"draft_answer": "I cannot answer this question based on the provided documents."}
    
    # 2. Format the context
    context_text = "\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    
    # 3. Build Failure Memory Injection
    failure_memory_prompt = ""
    if failed_drafts:
        failure_memory_prompt = "WARNING! You previously generated the following hallucinated answers. DO NOT REPEAT THEM:\n"
        for i, draft in enumerate(failed_drafts):
            failure_memory_prompt += f"- {draft}\n"
            
    llm = get_llm()
    
    # 4. The Strict Research Prompt
    template = """<|system|>
You are a highly reliable and strictly factual AI assistant.
Your ONLY source of truth is the Context provided below. 

CRITICAL RULES:
1. You MUST answer the user's question using ONLY the facts present in the Context.
2. If the Context does not contain the answer, you MUST reply EXACTLY with: "I cannot answer this question based on the provided documents."
3. DO NOT use your internal knowledge. DO NOT guess. DO NOT provide general information.

{failure_memory_prompt}

Context:
{context}<|end|>
<|user|>
Question: {input}<|end|>
<|assistant|>"""
    
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(
        context=context_text, 
        input=query, 
        failure_memory_prompt=failure_memory_prompt
    )
    
    # 5. Execute Inference
    print("  -> Generating grounded response...")
    draft = llm.invoke(formatted_prompt).strip()
    
    return {"draft_answer": draft}
