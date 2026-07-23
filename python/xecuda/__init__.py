"""
XeCUDA Python Integration Engine
Intel Arc GPU Compute — Qwythos-9B GGUF Inference
"""

__version__ = "2.0.0"

__all__ = [
    "XeCudaDevice",
    "WeightLoader",
    "Model",
    "benchmark_bandwidth",
]

from .device import XeCudaDevice
from .weight_loader import WeightLoader
from .model import Model
from .kernels import benchmark_bandwidth


def get_hardware_info():
    try:
        from .device import XeCudaDevice
        d = XeCudaDevice()
        info = d.info()
        d.shutdown()
        return {
            "gpu_name": info["gpu_name"],
            "vendor": info["vendor"],
            "driver": info["driver"],
            "compute_units": info["compute_units"],
            "total_memory_gb": info["total_memory_gb"],
            "max_alloc_gb": info["max_alloc_gb"],
            "xmx_supported": True,
            "architecture": "Xe2-LPG (Lunar Lake)"
        }
    except Exception:
        return {"error": "GPU not detected"}
