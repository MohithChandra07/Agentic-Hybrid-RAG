import os
import json
import csv
import time
from datetime import datetime
from evaluation.dataset import SAMPLE_DATASET
from evaluation.baseline import BaselineRAG
from evaluation.evaluators import LightweightEvaluator
from backend.graph import build_graph

class ExperimentRunner:
    """
    Orchestrates research experiments, running ablation studies 
    and benchmarking Baseline vs Agentic architectures.
    """
    def __init__(self):
        self.baseline = BaselineRAG()
        self.agentic_graph = build_graph()
        self.evaluator = LightweightEvaluator()
        
        # Ensure output directory exists
        self.output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_experiment(self, ablation_mode="full_comparison"):
        """
        Runs both Baseline and Agentic RAG on the dataset and logs all metrics.
        """
        results = []
        
        print(f"\n========== STARTING RESEARCH EXPERIMENT: {ablation_mode.upper()} ==========\n")
        
        for item in SAMPLE_DATASET:
            query = item["query"]
            print(f"\n[TESTING QUERY] {query}")
            
            # --- 1. RUN BASELINE RAG ---
            print("  -> Running Baseline RAG (Control)...")
            base_output = self.baseline.invoke(query)
            base_context = "\n".join([d.page_content for d in base_output["retrieved_docs"]])
            
            base_faith = self.evaluator.evaluate_faithfulness(base_context, base_output["final_answer"])
            base_rel = self.evaluator.evaluate_answer_relevance(query, base_output["final_answer"])
            base_context_rel = self.evaluator.evaluate_context_relevance(query, base_context)
            
            # --- 2. RUN AGENTIC RAG ---
            print("  -> Running Critic-Guided Agentic RAG (Experimental)...")
            start_time = time.time()
            agentic_state = self.agentic_graph.invoke({"original_query": query, "loop_count": 0})
            agentic_latency = time.time() - start_time
            
            ag_context = "\n".join([d.page_content for d in agentic_state.get("approved_context", [])])
            ag_answer = agentic_state.get("final_answer", "")
            
            ag_faith = self.evaluator.evaluate_faithfulness(ag_context, ag_answer)
            ag_rel = self.evaluator.evaluate_answer_relevance(query, ag_answer)
            ag_context_rel = self.evaluator.evaluate_context_relevance(query, ag_context)
            
            # Calculate Retry Effectiveness
            retry_history = agentic_state.get("retry_history", [])
            initial_relevance = retry_history[0]["relevance_score"] if retry_history else 0.0
            final_relevance = retry_history[-1]["relevance_score"] if retry_history else 0.0
            retry_effectiveness = final_relevance - initial_relevance if len(retry_history) > 1 else 0.0
            
            # --- 3. LOG METRICS ---
            result = {
                "query_id": item["id"],
                "query": query,
                "domain": item["category"],
                
                # Baseline Metrics
                "base_latency_sec": round(base_output["latency"], 2),
                "base_faithfulness": base_faith,
                "base_answer_relevance": base_rel,
                "base_context_relevance": base_context_rel,
                
                # Agentic Metrics
                "agentic_latency_sec": round(agentic_latency, 2),
                "agentic_faithfulness": ag_faith,
                "agentic_answer_relevance": ag_rel,
                "agentic_context_relevance": ag_context_rel,
                "agentic_loops_required": agentic_state.get("loop_count", 0),
                "agentic_retry_effectiveness": round(retry_effectiveness, 3),
                "agentic_failed_queries_memory": len(agentic_state.get("failed_queries", [])),
                "agentic_hybrid_confidence": agentic_state.get("retrieval_confidence_score", 0.0),
                "agentic_agreement_score": agentic_state.get("document_agreement_score", 1.0),
                
                # Generations (for human review)
                "base_answer": base_output["final_answer"],
                "agentic_answer": ag_answer
            }
            results.append(result)
            print(f"  [SCORES] Faithfulness - Base: {base_faith} | Agentic: {ag_faith}")
            print(f"  [SCORES] Context Rel - Base: {base_context_rel} | Agentic: {ag_context_rel}")
            
        self._save_results(results, ablation_mode)
        print("\n========== EXPERIMENT COMPLETE ==========")
        
    def _save_results(self, results, mode):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON (For detailed programmatic analysis)
        json_path = os.path.join(self.output_dir, f"experiment_{mode}_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        # Save CSV (For Excel/Graph generation)
        csv_path = os.path.join(self.output_dir, f"experiment_{mode}_{timestamp}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
                
        print(f"Metrics saved to {self.output_dir}/")

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_experiment(ablation_mode="full_comparison")
