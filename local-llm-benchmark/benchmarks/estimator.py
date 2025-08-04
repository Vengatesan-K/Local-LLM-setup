# Latency or throughput estimation utils
from transformers import AutoConfig
import torch

def estimate_model_requirements(model_name):
    """Estimate model requirements without loading it"""
    try:
        config = AutoConfig.from_pretrained(model_name)
        params = config.num_parameters() / 1e9  # in billions

        estimates = {
            "parameters": f"{params:.1f}B",
            "memory_estimates_gb": {
                "fp32": params * 4,
                "fp16": params * 2,
                "8bit": params * 1,
                "4bit": params * 0.5,
            },
            "recommended": {
                "gpu_memory_min": params * 2,
                "gpu_memory_recommended": params * 4,
                "cpu_ram_min": params * 8,
            },
        }

        return estimates
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import json

    models = ["meta-llama/Meta-Llama-3-8B", "Qwen/Qwen1.5-7B", "google/gemma-2b"]

    for model in models:
        print(f"\nEstimates for {model}:")
        print(json.dumps(estimate_model_requirements(model), indent=2))
