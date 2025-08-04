import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json


def cleanup_memory(model=None):
    """Clean up GPU memory"""
    if model is not None:
        del model
    torch.cuda.empty_cache()
    gc.collect()


def run_benchmark(
    model_name, prompt="Explain AI in simple terms", max_new_tokens=50, iterations=3
):
    results = {
        "model": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "iterations": iterations,
        "metrics": {},
    }

    model = None 
    tokenizer = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        _ = model.generate(**inputs, max_new_tokens=1)

        latencies = []
        tokens_per_sec = []
        memory_usages = []

        for _ in tqdm(range(iterations), desc=f"Benchmarking {model_name}"):
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()

            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

            latency = time.time() - start_time
            generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

            latencies.append(latency)
            tokens_per_sec.append(generated_tokens / latency)
            memory_usages.append(torch.cuda.max_memory_allocated() / (1024**2))

        results["metrics"] = {
            "avg_latency": sum(latencies) / len(latencies),
            "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec),
            "avg_tpm": (sum(tokens_per_sec) / len(tokens_per_sec)) * 60,
            "peak_memory_mb": max(memory_usages),
            "first_token_latency": latencies[0] / max_new_tokens,
        }

    except Exception as e:
        results["error"] = str(e)
    finally:
        cleanup_memory(model)
        if tokenizer is not None:
            del tokenizer

    return results


if __name__ == "__main__":
    import gc

    models = ["meta-llama/Meta-Llama-3-8B", "Qwen/Qwen1.5-7B", "google/gemma-2b"]

    for model in models:
        result = run_benchmark(model)
        print(f"\nBenchmark results for {model}:")
        print(json.dumps(result["metrics"], indent=2))
