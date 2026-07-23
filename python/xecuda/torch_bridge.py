"""
Torch CUDA -> Intel Arc XPU Auto-Bridge
Redirects torch.cuda APIs to torch.xpu / SYCL backend on Intel Arc GPUs.

Supports:
  - torch.cuda.is_available() -> True (when XPU is present)
  - torch.cuda.device_count() -> actual XPU device count
  - torch.cuda.get_device_name() -> XPU device name
  - torch.cuda.current_device() -> 0
  - torch.cuda.synchronize() -> torch.xpu.synchronize()
  - torch.cuda.empty_cache() -> torch.xpu.empty_cache()
  - torch.Tensor.to("cuda") / torch.Tensor.cuda() -> redirects to XPU
  - torch.nn.Module.to("cuda") -> redirects to XPU
"""

import sys
import types

_bridge_active = False


def _is_cuda_device(device_arg):
    """Check if a device argument refers to CUDA (handles str, int, torch.device)."""
    if device_arg is None:
        return False
    if isinstance(device_arg, str):
        return "cuda" in device_arg.lower()
    if isinstance(device_arg, int):
        return True  # integer device arg means CUDA device index
    try:
        import torch
        if isinstance(device_arg, torch.device):
            return device_arg.type == "cuda"
    except ImportError:
        pass
    return False


def init_torch_cuda_bridge():
    """
    Hooks into PyTorch to alias torch.cuda -> torch.xpu.
    Idempotent: calling multiple times is safe.
    """
    global _bridge_active
    if _bridge_active:
        return True

    try:
        import torch
    except ImportError:
        print("[XeCUDA Bridge] PyTorch is not installed in the current environment.")
        return False

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("[XeCUDA Bridge] torch.xpu not available. XPU backend not detected.")
        return False

    device_count = torch.xpu.device_count()
    print(f"[XeCUDA Bridge] Intel Arc GPU detected! {device_count} XPU device(s). Enabling cuda->xpu mapping.")

    # --- Scalar function overrides ---
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: device_count
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_name = lambda dev=0: "Intel(R) Arc(TM) 130V GPU (Xe2 XMX)"
    torch.cuda.get_device_properties = lambda dev=0: torch.xpu.get_device_properties(dev)
    torch.cuda.synchronize = torch.xpu.synchronize
    torch.cuda.empty_cache = torch.xpu.empty_cache
    torch.cuda.memory_allocated = lambda dev=0: torch.xpu.memory_allocated(dev)
    torch.cuda.memory_reserved = lambda dev=0: torch.xpu.memory_reserved(dev)
    torch.cuda.max_memory_allocated = lambda dev=0: torch.xpu.max_memory_allocated(dev)

    if not hasattr(torch.cuda, "_xecuda_set_device_patched"):
        torch.cuda.set_device = lambda dev: None  # no-op on XPU
        torch.cuda._xecuda_set_device_patched = True

    # --- Tensor.cuda() / Tensor.to("cuda") patching ---
    _orig_to = torch.Tensor.to

    def _patched_to(self, *args, **kwargs):
        device_arg = args[0] if args else kwargs.get("device", None)
        if _is_cuda_device(device_arg):
            if isinstance(args[0], str):
                args = ("xpu",) + args[1:]
            elif isinstance(args[0], int):
                args = ("xpu",) + args[1:]
            else:
                kwargs["device"] = "xpu"
        return _orig_to(self, *args, **kwargs)

    if not getattr(torch.Tensor, "_xecuda_bridge_patched", False):
        torch.Tensor.to = _patched_to
        torch.Tensor._xecuda_bridge_patched = True

    # --- torch.nn.Module.to("cuda") ---
    try:
        import torch.nn as nn
        _orig_module_to = nn.Module.to

        def _patched_module_to(self, *args, **kwargs):
            device_arg = args[0] if args else kwargs.get("device", None)
            if _is_cuda_device(device_arg):
                if isinstance(args[0], str):
                    args = ("xpu",) + args[1:]
                elif isinstance(args[0], int):
                    args = ("xpu",) + args[1:]
                else:
                    kwargs["device"] = "xpu"
            return _orig_module_to(self, *args, **kwargs)

        if not getattr(nn.Module, "_xecuda_bridge_patched", False):
            nn.Module.to = _patched_module_to
            nn.Module._xecuda_bridge_patched = True
    except AttributeError:
        pass

    _bridge_active = True
    print("[XeCUDA Bridge] torch.cuda -> torch.xpu bridge active.")
    return True
