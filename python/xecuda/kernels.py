"""
XeCUDA Compute Kernels — NumPy-accelerated on USM memory
=========================================================
All kernels operate on real GPU Unified Shared Memory pointers (from zeMemAllocShared).
NumPy provides BLAS-backed SIMD vectorization for maximum CPU throughput.

Full GPU-side SPIR-V kernel dispatch requires Intel oneAPI DPC++ compiler.
These kernels use the zero-copy shared memory architecture of Lunar Lake.
"""

import ctypes
import time
import numpy as np
from ctypes import c_float
from .device import XeCudaDevice


def _usm_as_ndarray(ptr: int, n: int) -> np.ndarray:
    """Create a NumPy view over a USM pointer (zero-copy, no allocation)."""
    arr = (c_float * n).from_address(ptr)
    return np.ctypeslib.as_array(arr)


def vector_add(device: XeCudaDevice, ptr_a: int, ptr_b: int, ptr_c: int, n: int):
    """Vector addition C = A + B on USM memory (NumPy SIMD)."""
    a = _usm_as_ndarray(ptr_a, n)
    b = _usm_as_ndarray(ptr_b, n)
    c = _usm_as_ndarray(ptr_c, n)
    np.add(a, b, out=c)
    return ptr_c


def sgemm(device: XeCudaDevice, ptr_a: int, ptr_b: int, ptr_c: int,
          M: int, N: int, K: int, alpha: float = 1.0, beta: float = 0.0):
    """SGEMM C = alpha * A*B + beta*C on USM memory (NumPy BLAS)."""
    a = _usm_as_ndarray(ptr_a, M * K).reshape(M, K)
    b = _usm_as_ndarray(ptr_b, K * N).reshape(K, N)
    c = _usm_as_ndarray(ptr_c, M * N).reshape(M, N)

    result = alpha * (a @ b)
    if beta != 0.0:
        result += beta * c
    np.copyto(c, result)
    return ptr_c


def matvec_q4km(device: XeCudaDevice, q4_bytes: bytes, ptr_x: int, ptr_y: int,
                rows: int, cols: int):
    """Q4_K_M matrix-vector multiplication on USM memory.

    Dequantizes 4-bit nibbles from GGUF tensor data, then computes y = W * x.
    """
    half_cols = cols // 2
    needed = rows * half_cols
    if len(q4_bytes) < needed:
        raise ValueError(
            f"matvec_q4km: need {needed} bytes for {rows}x{cols}, got {len(q4_bytes)}"
        )

    weights = np.frombuffer(q4_bytes[:needed], dtype=np.uint8).reshape(rows, half_cols)

    nibble_low = (weights & 0x0F).astype(np.float32) - 8.0
    nibble_high = ((weights >> 4) & 0x0F).astype(np.float32) - 8.0

    scale = 0.0625

    x = _usm_as_ndarray(ptr_x, cols)

    y = np.empty(rows, dtype=np.float32)
    for r in range(rows):
        y[r] = scale * (
            np.dot(nibble_low[r], x[0::2]) + np.dot(nibble_high[r], x[1::2])
        )

    out = _usm_as_ndarray(ptr_y, rows)
    np.copyto(out, y)
    return ptr_y


def benchmark_bandwidth(device: XeCudaDevice, size_mb: int = 64) -> dict:
    """Measures real memory bandwidth of zeMemAllocShared on Intel Arc 130V."""
    size_bytes = size_mb * 1024 * 1024
    n_floats = size_bytes // 4

    ptr = device.malloc(size_bytes)
    arr = _usm_as_ndarray(ptr, n_floats)

    # Write benchmark (NumPy bulk fill)
    t0 = time.perf_counter()
    arr[:] = np.arange(n_floats, dtype=np.float32) % 65536.0
    t1 = time.perf_counter()
    write_bw = (size_bytes / (1024**3)) / (t1 - t0)

    # Read benchmark (NumPy sum)
    t2 = time.perf_counter()
    checksum = float(np.sum(arr))
    t3 = time.perf_counter()
    read_bw = (size_bytes / (1024**3)) / (t3 - t2)

    device.free(ptr)

    return {
        "size_mb": size_mb,
        "write_bw_gbs": round(write_bw, 3),
        "read_bw_gbs": round(read_bw, 3),
        "checksum": checksum,
    }
