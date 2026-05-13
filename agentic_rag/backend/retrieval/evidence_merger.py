from langchain_core.documents import Document
from backend.core.dependencies import get_llm
from langchain_core.prompts import PromptTemplate

def merge_evidence(queries: list[str], docs: list[Document]) -> list[Document]:
    """Intelligently merges and filters retrieved evidence for complex multi-hop sub-queries."""
    # If it's a simple query or we don't have enough docs, return them unmodified
    if len(queries) <= 1 or not docs:
        return docs 
        
    print("--- EVIDENCE MERGER: Synthesizing documents from multiple sub-queries ---")
    
    llm = get_llm()
    context_text = "\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    
    template = """<|system|>
You are an Evidence Merger Agent. The user asked multiple sub-queries which requires multi-hop reasoning.
Read the provided raw documents. Synthesize a highly concentrated, factual summary that extracts ONLY the evidence necessary to answer all sub-queries.
Do NOT attempt to answer the query directly. Just extract and merge the raw facts into a clean block of context.
<|user|>
Sub-queries: {queries}
Documents:
{context}<|end|>
<|assistant|>"""
    
    prompt = PromptTemplate.from_template(template)
    response = llm.invoke(prompt.format(queries=" | ".join(queries), context=context_text)).strip()
    
    # Return the merged evidence as a single high-density document
    merged_doc = Document(
        page_content=response, 
        metadata={"source": "Merged_Evidence_Agent", "rrf_score": 1.0}
    )
    
    print("  -> Multi-hop evidence successfully merged.")
    return [merged_doc]
