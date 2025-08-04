# Benchmark for batched inference
from .basic_benchmark import run_benchmark
from tqdm import tqdm
import json


def run_batch_benchmark(model_name, batch_sizes=[1, 2, 4], **kwargs):
    results = []
    for batch_size in tqdm(batch_sizes, desc="Batch sizes"):
        current_kwargs = kwargs.copy()
        current_kwargs["prompt"] = [kwargs.get("prompt", "Explain AI")] * batch_size
        result = run_benchmark(model_name, **current_kwargs)
        result["batch_size"] = batch_size
        results.append(result)
    return results


if __name__ == "__main__":
    results = run_batch_benchmark("google/gemma-2b")
    print(json.dumps(results, indent=2))
