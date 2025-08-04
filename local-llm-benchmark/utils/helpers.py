# Shared utility functions
import torch
import gc
from tabulate import tabulate

def cleanup_memory(model=None):
    """Clean up GPU memory"""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_results(results):
    """Pretty print benchmark results"""
    

    table = []
    for result in results:
        if "error" in result:
            table.append([result["model"], "ERROR", result["error"]])
            continue

        metrics = result["metrics"]
        table.append(
            [
                result["model"],
                f"{metrics['avg_tpm']:.1f}",
                f"{metrics['avg_latency']:.2f}s",
                f"{metrics['peak_memory_mb']:.1f}MB",
                result["device"],
            ]
        )

    print(
        tabulate(
            table,
            headers=["Model", "TPM", "Latency", "Memory", "Device"],
            tablefmt="github",
        )
    )
