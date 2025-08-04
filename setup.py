import os

# Directory and file structure
structure = {
    "local-llm-benchmark": {
        "README.md": "# Local LLM Benchmark\n\nBenchmarking quantized, batch, and basic local LLMs.",
        "LICENSE": "MIT License\n\nCopyright (c) 2025",
        "requirements.txt": "# Add your Python dependencies here\ntransformers\naccelerate\n",
        "benchmarks": {
            "__init__.py": "",
            "hardware_profiler.py": "# Utility to detect CPU/GPU info\n",
            "basic_benchmark.py": "# Benchmark for basic inference\n",
            "quantized_benchmark.py": "# Benchmark for quantized models\n",
            "batch_benchmark.py": "# Benchmark for batched inference\n",
            "estimator.py": "# Latency or throughput estimation utils\n",
        },
        "utils": {"__init__.py": "", "helpers.py": "# Shared utility functions\n"},
    }
}


def create_structure(base_path, tree):
    for name, content in tree.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Created: {path}")


if __name__ == "__main__":
    create_structure(".", structure)
    print("\n✅ Project scaffold 'local-llm-benchmark' created successfully.")
