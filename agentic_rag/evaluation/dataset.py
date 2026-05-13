import json
import os

# Define the structured dataset path
DATASET_PATH = os.path.join(os.path.dirname(__file__), "benchmark_samples.json")

def load_benchmark_dataset():
    """Loads the structured benchmark JSON for rigorous evaluation."""
    if not os.path.exists(DATASET_PATH):
        print(f"Warning: Dataset not found at {DATASET_PATH}. Returning empty list.")
        return []
        
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Global access point for the runner
SAMPLE_DATASET = load_benchmark_dataset()
