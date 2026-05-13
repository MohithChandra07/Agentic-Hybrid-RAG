import re
import time
from backend.core.dependencies import get_llm
from langchain_core.prompts import PromptTemplate

class LightweightEvaluator:
    """
    Research-grade LLM-as-a-Judge Evaluator.
    Uses the local Phi-3 model to score generation and retrieval quality.
    Equivalent to lightweight RAGAS metrics but optimized for local inference.
    """
    def __init__(self):
        self.llm = get_llm()
        
    def evaluate_faithfulness(self, context: str, answer: str) -> float:
        """Measures Hallucination: Is the answer entirely grounded in the context?"""
        if not context.strip():
            # If no context but an answer was generated, that's a 100% hallucination
            return 0.0 if answer.strip() and "do not have" not in answer.lower() else 1.0
            
        template = """<|system|>
You are a strict grading metric. Read the Answer and the Context.
Does the Answer contain ANY information that cannot be directly proven by the Context?
If it contains outside information (hallucination), output SCORE: 0.0
If it is perfectly faithful to the context, output SCORE: 1.0
Context: {context}
<|user|>
Answer: {answer}
<|assistant|>"""
        prompt = PromptTemplate.from_template(template)
        result = self.llm.invoke(prompt.format(context=context, answer=answer))
        return self._extract_score(result)

    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Measures Relevance: Does the answer actually address the question?"""
        template = """<|system|>
You are a strict grading metric. Read the Query and the Answer.
Does the Answer directly address the user's query? 
If yes, output SCORE: 1.0. If partially, output SCORE: 0.5. If irrelevant, output SCORE: 0.0.
<|user|>
Query: {query}
Answer: {answer}
<|assistant|>"""
        prompt = PromptTemplate.from_template(template)
        result = self.llm.invoke(prompt.format(query=query, answer=answer))
        return self._extract_score(result)

    def evaluate_context_relevance(self, query: str, context: str) -> float:
        """Measures Retrieval Quality: Does the context contain the answer?"""
        if not context.strip(): return 0.0
        
        template = """<|system|>
You are a strict grading metric. Read the Query and the Context.
Does the Context contain the exact factual answer to the Query?
If it contains the complete answer, output SCORE: 1.0.
If it contains partial or related information, output SCORE: 0.5.
If it is completely irrelevant, output SCORE: 0.0.
<|user|>
Query: {query}
Context: {context}
<|assistant|>"""
        prompt = PromptTemplate.from_template(template)
        result = self.llm.invoke(prompt.format(query=query, context=context))
        return self._extract_score(result)

    def _extract_score(self, text: str) -> float:
        match = re.search(r"SCORE:\s*([0-9.]+)", text.upper())
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        return 0.0
