"""
XeCUDA Python Integration Engine
Intel Arc GPU Compute — OpenCL 3.0 backend (Intel NEO), dependency-free core.

The GPU core (device + kernels) is pure ctypes / OpenCL and imports fine even
when numpy / torch are broken or absent. The optional model-loading pieces
(WeightLoader, Model) import numpy/gguf lazily, only if you use them.
"""

__version__ = "2.1.0"

# ── Eager, dependency-free core (ctypes / OpenCL only) ──────────────────
from .device import XeCudaDevice
from . import kernels            # sets the package attribute `kernels`

__all__ = [
    "XeCudaDevice",
    "kernels",
    "get_hardware_info",
]


def __getattr__(name):
    """Lazy-load the optional numpy-heavy modules only when requested."""
    if name == "WeightLoader":
        from . import weight_loader
        return weight_loader.WeightLoader
    if name == "Model":
        from . import model
        return model.Model
    if name == "init_torch_cuda_bridge":
        from .torch_bridge import init_torch_cuda_bridge
        return init_torch_cuda_bridge
    raise AttributeError(f"module 'xecuda' has no attribute {name!r}")


def get_hardware_info():
    try:
        d = XeCudaDevice()
        info = d.info()
        d.shutdown()
        return info
    except Exception as e:
        return {"error": f"GPU not detected: {e}"}