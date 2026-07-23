"""
XeCUDA GPU Kernels — Real OpenCL dispatch on Intel Arc 130V
===========================================================
ALL computation runs on the actual GPU via OpenCL kernel dispatch.
No NumPy fallback. No CPU simulation. Real GPU compute.
"""

import time
import ctypes
from .device import XeCudaDevice, _P


# ─── OpenCL kernel sources ───────────────────────────────────────────

VEC_ADD_SRC = b"""
__kernel void vector_add(
    __global const float* A, __global const float* B,
    __global float* C, const int N)
{
    int i = get_global_id(0);
    if (i < N) C[i] = A[i] + B[i];
}
"""

SGEMM_SRC = b"""
__kernel void sgemm(
    const int M, const int N, const int K,
    __global const float* A, __global const float* B,
    __global float* C,
    const float alpha, const float beta)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
"""

MATVEC_Q4_SRC = b"""
__kernel void matvec_q4km(
    __global const uchar* q4_weights,
    __global const float* x,
    __global float* y,
    const int rows, const int cols)
{
    int r = get_global_id(0);
    if (r >= rows) return;
    int half_cols = cols / 2;
    float sum = 0.0f;
    int row_offset = r * half_cols;
    for (int c = 0; c < half_cols; c++) {
        uchar byte_val = q4_weights[row_offset + c];
        int nibble_low = (byte_val & 0x0F) - 8;
        int nibble_high = ((byte_val >> 4) & 0x0F) - 8;
        int col = c * 2;
        if (col < cols)
            sum += (float)nibble_low * 0.0625f * x[col];
        if (col + 1 < cols)
            sum += (float)nibble_high * 0.0625f * x[col + 1];
    }
    y[r] = sum;
}
"""


def vector_add(device: XeCudaDevice, ptr_a: int, ptr_b: int, ptr_c: int, n: int):
    """Vector addition C = A + B — dispatched to Intel Arc GPU via OpenCL."""
    k = device.build_kernel("vec_add", VEC_ADD_SRC, b"vector_add")
    bufA = ctypes.c_void_p(ptr_a)
    bufB = ctypes.c_void_p(ptr_b)
    bufC = ctypes.c_void_p(ptr_c)
    n_val = ctypes.c_int32(n)
    device.enqueue_kernel(k, (n + 255) // 256 * 256, 256, [bufA, bufB, bufC, n_val])
    device.finish()
    return ptr_c


def sgemm(device: XeCudaDevice, ptr_a: int, ptr_b: int, ptr_c: int,
          M: int, N: int, K: int, alpha: float = 1.0, beta: float = 0.0):
    """SGEMM C = alpha*A*B + beta*C — dispatched to Intel Arc GPU via OpenCL."""
    k = device.build_kernel("sgemm", SGEMM_SRC, b"sgemm")
    bufA = ctypes.c_void_p(ptr_a)
    bufB = ctypes.c_void_p(ptr_b)
    bufC = ctypes.c_void_p(ptr_c)
    args = [
        ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K),
        bufA, bufB, bufC,
        ctypes.c_float(alpha), ctypes.c_float(beta)
    ]
    device.enqueue_kernel(k, max(M, N), 16, args)
    device.finish()
    return ptr_c


def matvec_q4km(device: XeCudaDevice, q4_bytes: bytes, ptr_x: int, ptr_y: int,
                rows: int, cols: int):
    """Q4_K_M matrix-vector multiply — dispatched to Intel Arc GPU via OpenCL."""
    half_cols = cols // 2
    needed = rows * half_cols
    if len(q4_bytes) < needed:
        raise ValueError(f"matvec_q4km: need {needed} bytes, got {len(q4_bytes)}")

    ptr_q4 = device.malloc(needed)
    device.write_buffer(ptr_q4, (ctypes.c_ubyte * needed)(*q4_bytes[:needed]), needed)

    k = device.build_kernel("matvec_q4km", MATVEC_Q4_SRC, b"matvec_q4km")
    buf_q4 = ctypes.c_void_p(ptr_q4)
    buf_x = ctypes.c_void_p(ptr_x)
    buf_y = ctypes.c_void_p(ptr_y)
    device.enqueue_kernel(k, (rows + 255) // 256 * 256, 256,
                          [buf_q4, buf_x, buf_y, ctypes.c_int32(rows), ctypes.c_int32(cols)])
    device.finish()
    device.free(ptr_q4)
    return ptr_y


def benchmark_bandwidth(device: XeCudaDevice, size_mb: int = 64) -> dict:
    """Measures real GPU memory bandwidth via OpenCL read/write."""
    size_bytes = size_mb * 1024 * 1024
    n_floats = size_bytes // 4

    buf = device.malloc(size_bytes)
    src = (ctypes.c_float * n_floats)(*[float(i % 65536) for i in range(n_floats)])

    t0 = time.perf_counter()
    device.write_buffer(buf, ctypes.byref(src), size_bytes)
    device.finish()
    t1 = time.perf_counter()
    write_bw = (size_bytes / (1024**3)) / (t1 - t0)

    out = (ctypes.c_float * n_floats)()
    t2 = time.perf_counter()
    device.read_buffer(buf, ctypes.byref(out), size_bytes)
    device.finish()
    t3 = time.perf_counter()
    read_bw = (size_bytes / (1024**3)) / (t3 - t2)

    device.free(buf)
    return {
        "size_mb": size_mb,
        "write_bw_gbs": round(write_bw, 3),
        "read_bw_gbs": round(read_bw, 3),
        "checksum": float(sum(out[i] for i in range(min(100, n_floats)))),
    }
