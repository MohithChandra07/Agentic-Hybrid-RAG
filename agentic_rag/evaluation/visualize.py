import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Ensure seaborn style for publication readiness
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def load_latest_experiment_data(results_dir: str) -> pd.DataFrame:
    """Loads the most recently created CSV file from the results directory."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"Loading data from: {latest_file}")
    return pd.read_csv(latest_file)

def plot_faithfulness_comparison(df: pd.DataFrame, output_dir: str):
    """Generates a bar chart comparing Baseline vs Agentic Faithfulness (Hallucination Reduction)."""
    plt.figure(figsize=(8, 6))
    
    means = [df['base_faithfulness'].mean(), df['agentic_faithfulness'].mean()]
    labels = ['Baseline RAG', 'Agentic RAG']
    
    ax = sns.barplot(x=labels, y=means, hue=labels, palette=['#e74c3c', '#2ecc71'], legend=False)
    
    plt.title('Faithfulness Comparison (Hallucination Reduction)', pad=20, fontweight='bold')
    plt.ylabel('Average Faithfulness Score (Higher is better)')
    plt.ylim(0, 1.1)
    
    # Add value labels on top of bars
    for i, v in enumerate(means):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'faithfulness_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_latency_comparison(df: pd.DataFrame, output_dir: str):
    """Generates a graph comparing latency tradeoffs."""
    plt.figure(figsize=(8, 6))
    
    means = [df['base_latency_sec'].mean(), df['agentic_latency_sec'].mean()]
    labels = ['Baseline RAG', 'Agentic RAG']
    
    ax = sns.barplot(x=labels, y=means, hue=labels, palette=['#3498db', '#e67e22'], legend=False)
    
    plt.title('Latency Tradeoff Comparison', pad=20, fontweight='bold')
    plt.ylabel('Average Latency (Seconds)')
    
    for i, v in enumerate(means):
        plt.text(i, v + 0.1, f'{v:.2f}s', ha='center', fontweight='bold')
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'latency_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_retrieval_precision(df: pd.DataFrame, output_dir: str):
    """Generates a comparison of Context Relevance."""
    if 'base_context_relevance' not in df.columns or 'agentic_context_relevance' not in df.columns:
        print("Skipping Retrieval Precision: Missing columns.")
        return
        
    plt.figure(figsize=(8, 6))
    
    means = [df['base_context_relevance'].mean(), df['agentic_context_relevance'].mean()]
    labels = ['Baseline RAG', 'Agentic RAG']
    
    ax = sns.barplot(x=labels, y=means, hue=labels, palette=['#9b59b6', '#1abc9c'], legend=False)
    
    plt.title('Retrieval Precision (Context Relevance)', pad=20, fontweight='bold')
    plt.ylabel('Average Context Relevance Score')
    plt.ylim(0, 1.1)
    
    for i, v in enumerate(means):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'retrieval_precision.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_retry_effectiveness(df: pd.DataFrame, output_dir: str):
    """Generates a graph tracking the effectiveness of retry loops."""
    if 'agentic_retry_effectiveness' not in df.columns:
        print("Skipping Retry Effectiveness: Missing columns.")
        return
        
    plt.figure(figsize=(8, 6))
    
    sns.histplot(df['agentic_retry_effectiveness'], bins=10, kde=True, color='#34495e')
    
    plt.title('Distribution of Retry Loop Effectiveness', pad=20, fontweight='bold')
    plt.xlabel('Relevance Improvement Score (Final - Initial)')
    plt.ylabel('Frequency')
    
    plt.axvline(x=0, color='r', linestyle='--', label='No Improvement')
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'retry_effectiveness.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_all_visualizations():
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(eval_dir, "results")
    graphs_dir = os.path.join(eval_dir, "graphs")
    
    os.makedirs(graphs_dir, exist_ok=True)
    
    try:
        df = load_latest_experiment_data(results_dir)
        
        print(f"\nGenerating publication-ready visualizations to {graphs_dir}...\n")
        plot_faithfulness_comparison(df, graphs_dir)
        plot_latency_comparison(df, graphs_dir)
        plot_retrieval_precision(df, graphs_dir)
        plot_retry_effectiveness(df, graphs_dir)
        
        print("\nAll visualizations completed successfully!")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    generate_all_visualizations()
