"""
XeCUDA GGUF Weight Loader — Upload Quantized Weights to GPU
============================================================
Uses the gguf Python library to read tensor data correctly,
then uploads raw quantized bytes directly to OpenCL GPU buffers.
No dequantization on CPU — all dequant happens inside GPU kernels.
"""

import sys
import os
import ctypes
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from gguf import GGUFReader

GGUF_PATH = os.environ.get(
    "XECUDA_MODEL_PATH",
    os.path.expanduser(
        r"~\.lmstudio\models\empero-ai\Qwythos-9B-Claude-Mythos-5-1M-GGUF"
        r"\Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"
    ),
)

# Quantization type -> bytes per 256-value block
QTYPE_BLOCK_BYTES = {
    0: 1024,   # F32: 256 * 4
    12: 144,   # Q4_K
    14: 210,   # Q6_K
}

# Quantization type -> name
QTYPE_NAMES = {0: "F32", 12: "Q4_K", 14: "Q6_K"}


def compute_n_bytes(tensor):
    """Compute exact byte count for a tensor based on its quantization type."""
    n_elements = 1
    for s in tensor.shape:
        n_elements *= int(s)
    qt = int(tensor.tensor_type)
    if qt == 0:
        return n_elements * 4
    elif qt in QTYPE_BLOCK_BYTES:
        n_blocks = n_elements // 256
        if n_blocks == 0:
            print(f"  WARNING: tensor {tensor.name} has {n_elements} elements "
                  f"(< 256) with Qtype={qt}, rounding up to 1 block")
            n_blocks = 1
        return n_blocks * QTYPE_BLOCK_BYTES[qt]
    else:
        raise ValueError(f"Unknown quantization type {qt} for tensor {tensor.name}")


class WeightLoader:
    """Load all model weights from GGUF file and upload to GPU."""

    def __init__(self, device, gguf_path=None):
        self.device = device
        self.gguf_path = gguf_path or GGUF_PATH
        self.reader = None
        self.gpu_buffers = {}
        self.tensor_info = {}
        self.total_bytes = 0

    def load(self, verbose=True):
        t0 = time.perf_counter()
        if verbose:
            print(f"[WeightLoader] Reading GGUF: {os.path.basename(self.gguf_path)}")
        self.reader = GGUFReader(self.gguf_path)

        # Skip tokenizer vocab arrays — only read weight tensors
        weight_tensors = []
        skip_prefixes = ("tokenizer", "charlist")
        for t in self.reader.tensors:
            name = t.name
            if any(name.startswith(p) for p in skip_prefixes):
                continue
            if name.endswith(".weight") or name.endswith(".bias"):
                weight_tensors.append(t)
            elif name in ("output_norm.weight",):
                weight_tensors.append(t)
            # Include SSM scalar params
            elif ".ssm_" in name and len(t.shape) == 1:
                weight_tensors.append(t)

        if verbose:
            print(f"[WeightLoader] Found {len(weight_tensors)} weight tensors")

        # Load each tensor to GPU
        for i, t in enumerate(weight_tensors):
            n_bytes = compute_n_bytes(t)
            raw = t.data.tobytes()[:n_bytes]

            gpu_buf = self.device.malloc(n_bytes)
            host_buf = (ctypes.c_ubyte * n_bytes)()
            ctypes.memmove(host_buf, raw, n_bytes)
            self.device.write_buffer(gpu_buf, host_buf, n_bytes)

            self.gpu_buffers[t.name] = gpu_buf
            self.tensor_info[t.name] = {
                "shape": [int(s) for s in t.shape],
                "type": int(t.tensor_type),
                "n_bytes": n_bytes,
                "n_elements": self._count_elements(t),
            }
            self.total_bytes += n_bytes

            if verbose and (i % 50 == 0 or i == len(weight_tensors) - 1):
                gb = self.total_bytes / (1024**3)
                print(f"  [{i+1}/{len(weight_tensors)}] loaded {t.name} "
                      f"{[int(s) for s in t.shape]} "
                      f"{QTYPE_NAMES.get(int(t.tensor_type), '?')} "
                      f"({n_bytes/1024/1024:.1f} MB) total={gb:.2f} GB")

        t1 = time.perf_counter()
        if verbose:
            print(f"[WeightLoader] Done: {len(self.gpu_buffers)} tensors, "
                  f"{self.total_bytes/1e9:.2f} GB in {t1-t0:.1f}s")
        return self

    def get(self, name):
        """Get GPU buffer handle for a tensor by name."""
        return self.gpu_buffers.get(name)

    def info(self, name):
        """Get metadata for a tensor by name."""
        return self.tensor_info.get(name)

    def qtype(self, name):
        """Get quantization type for a tensor by name (12=Q4_K, 14=Q6_K, 0=F32)."""
        info = self.tensor_info.get(name)
        return info["type"] if info else -1

    def _count_elements(self, tensor):
        n = 1
        for s in tensor.shape:
            n *= int(s)
        return n

    def shutdown(self):
        for buf in self.gpu_buffers.values():
            try:
                self.device.free(buf)
            except Exception:
                pass
        self.gpu_buffers.clear()
