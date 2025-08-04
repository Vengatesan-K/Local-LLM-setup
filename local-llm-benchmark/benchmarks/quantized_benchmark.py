from transformers import BitsAndBytesConfig
from .basic_benchmark import run_benchmark
import torch

QUANT_CONFIGS = {
    "4bit": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    ),
    "8bit": BitsAndBytesConfig(
        load_in_8bit=True,
    ),
}


def run_quantized_benchmark(model_name, quant_type="4bit", **kwargs):
    if quant_type not in QUANT_CONFIGS:
        raise ValueError(f"Unsupported quantization type: {quant_type}")

    kwargs["quantization_config"] = QUANT_CONFIGS[quant_type]
    return run_benchmark(model_name, **kwargs)


if __name__ == "__main__":
    import json

    results = run_quantized_benchmark("google/gemma-2b", quant_type="4bit")
    print(json.dumps(results, indent=2))
