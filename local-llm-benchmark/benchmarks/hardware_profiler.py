# Utility to detect CPU/GPU info
import torch
import psutil


def get_hardware_info():
    """Collect comprehensive hardware information"""
    info = {
        "cpu": {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "min_freq": psutil.cpu_freq().min,
            "max_freq": psutil.cpu_freq().max,
            "architecture": torch.__config__.parallel_info(),
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count(),
            "devices": [],
        },
        "memory": {
            "total": round(psutil.virtual_memory().total / (1024**3), 2),
            "available": round(psutil.virtual_memory().available / (1024**3), 2),
        },
        "swap": {
            "total": round(psutil.swap_memory().total / (1024**3), 2),
            "used": round(psutil.swap_memory().used / (1024**3), 2),
        },
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpu"]["devices"].append(
                {
                    "name": props.name,
                    "memory": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessors": props.multi_processor_count,
                }
            )

    return info


if __name__ == "__main__":
    import json

    print(json.dumps(get_hardware_info(), indent=2))
