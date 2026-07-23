"""
XeCUDA Python Integration Engine
Intel Arc GPU Compute & CUDA Migration Suite (Intel Arc 130V Lunar Lake Optimized)
"""

__version__ = "1.0.0"

__all__ = [
    "XeCudaDevice",
    "IntelArcInferenceEngine",
    "init_torch_cuda_bridge",
    "vector_add",
    "sgemm",
    "matvec_q4km",
    "benchmark_bandwidth",
    "Tensor",
    "Adam",
]


def get_hardware_info():
    """Return detected Intel GPU hardware specifications via OpenCL."""
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


from .torch_bridge import init_torch_cuda_bridge
from .inference import IntelArcInferenceEngine
from .device import XeCudaDevice
from .kernels import vector_add, sgemm, matvec_q4km, benchmark_bandwidth
from .autograd import Tensor, Adam
