import sys
import os

# Add the agentic_rag folder to the system path so Python can find the new architecture
AGENTIC_RAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_rag")
sys.path.append(AGENTIC_RAG_DIR)

# Import the new Agentic state machine graph
# pyrefly: ignore [missing-import]
from backend.graph import build_graph
# pyrefly: ignore [missing-import]
from backend.core.logger import log_agentic_state
import time

def main():
    print("\n" + "="*70)
    print(" 🤖 CRITIC-GUIDED AGENTIC HYBRID RAG SYSTEM IS ONLINE 🤖 ")
    print("="*70)
    print(" ✓ Hybrid Search Enabled (ChromaDB + BM25)")
    print(" ✓ Context Critic Enabled (Hallucination Prevention)")
    print(" ✓ Web Retriever Enabled (Internet Fallback for Out-of-Domain)")
    print("-" * 70)
    print("Type 'exit' or 'quit' to terminate the system.\n")
    
    # Initialize the powerful LangGraph multi-agent system
    print("Initializing Agents and loading models...")
    app = build_graph()
    print("System Ready!\n")
    
    while True:
        try:
            query = input("\n👤 You: ")
            if query.lower() in ['exit', 'quit']:
                print("\nShutting down the Agentic RAG. Goodbye!\n")
                break
            if not query.strip():
                continue
                
            initial_state = {
                "original_query": query,
                "loop_count": 0
            }
            
            print("\n⚙️  Agent is thinking...")
            
            start_time = time.time()
            final_state = initial_state
            
            # 1. Stream the nodes to show the AI's internal thought process
            for output in app.stream(initial_state):
                for key, state in output.items():
                    final_state = state
                    
            latency = time.time() - start_time
            
            # 2. Log structured telemetry
            log_agentic_state(final_state, latency)
            
            # 3. Print the final answer beautifully
            
            # 3. Print the final answer beautifully
            print("\n" + "="*70)
            print("💡 FINAL ANSWER:")
            print("-" * 70)
            print(final_state.get("final_answer", "Error: No answer generated."))
            print("="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nShutting down the Agentic RAG. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}\n")

if __name__ == "__main__":
    main()