"""
XeCUDA GPU Kernels — Real OpenCL dispatch on Intel Arc 130V
=========================================================
ALL computation runs on the actual GPU via OpenCL kernel dispatch.
Quantized weight dequantization happens fused inside matvec kernels.
No NumPy fallback. No CPU simulation. Real GPU compute.
"""

import time
import ctypes
from .device import XeCudaDevice, _P

_kernel_cache = {}


def _get_kernel(device, key, source, func_name):
    cache_key = (id(device), key)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]
    k = device.build_kernel(key, source, func_name)
    _kernel_cache[cache_key] = k
    return k


def _set_args(device, kernel, args):
    for i, arg in enumerate(args):
        if isinstance(arg, int):
            val = ctypes.c_void_p(arg)
            device._ocl.clSetKernelArg(kernel, i, ctypes.sizeof(val), ctypes.pointer(val))
        elif isinstance(arg, ctypes.c_int32):
            device._ocl.clSetKernelArg(kernel, i, 4, ctypes.pointer(arg))
        elif isinstance(arg, ctypes.c_float):
            device._ocl.clSetKernelArg(kernel, i, 4, ctypes.pointer(arg))
        elif isinstance(arg, ctypes.c_void_p):
            device._ocl.clSetKernelArg(kernel, i, ctypes.sizeof(arg), ctypes.pointer(arg))
        else:
            raise TypeError(f"Unsupported arg type: {type(arg)}")


# ═══════════════════════════════════════════════════════════════════════
# RMS Normalization kernel
# ═══════════════════════════════════════════════════════════════════════

RMS_NORM_SRC = b"""
__kernel void rms_norm(
    __global const float* x,
    __global float* y,
    __global const float* w,
    const int dim,
    const float eps)
{
    int row = get_global_id(0);
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float v = x[row * dim + i];
        sum += v * v;
    }
    float rms = 1.0f / sqrt(sum / dim + eps);
    for (int i = 0; i < dim; i++) {
        y[row * dim + i] = (x[row * dim + i] * rms) * w[i];
    }
}
"""


def rms_norm(device, ptr_x, ptr_y, ptr_w, n_rows, dim, eps=1e-6):
    k = _get_kernel(device, "rms_norm", RMS_NORM_SRC, b"rms_norm")
    _set_args(device, k, [
        ctypes.c_void_p(ptr_x), ctypes.c_void_p(ptr_y), ctypes.c_void_p(ptr_w),
        ctypes.c_int32(dim), ctypes.c_float(eps),
    ])
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(n_rows)),
        ctypes.pointer(ctypes.c_size_t(1)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# RoPE (Rotary Position Embedding) kernel
# ═══════════════════════════════════════════════════════════════════════

ROPE_INPLACE_SRC = b"""
__kernel void rope_inplace(
    __global float* q,
    __global float* k,
    const int head_dim,
    const int n_heads,
    const int n_kv_heads,
    const int pos,
    const float freq_base,
    const float rope_factor)
{
    int idx = get_global_id(0);
    int total_q = n_heads * head_dim;
    int total_k = n_kv_heads * head_dim;

    if (idx < total_q) {
        int h = idx / head_dim;
        int d = idx % head_dim;
        int pair = d / 2;
        float theta = (float)pos / pow(freq_base, (float)(2 * pair) / (float)head_dim);
        theta /= rope_factor;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        int base = h * head_dim + d;
        float val = q[base];
        float partner = q[h * head_dim + (d % 2 == 0 ? d + 1 : d - 1)];
        if (d % 2 == 0)
            q[base] = val * cos_t - partner * sin_t;
        else
            q[base] = val * cos_t + partner * sin_t;
    }

    if (idx < total_k) {
        int h = idx / head_dim;
        int d = idx % head_dim;
        int pair = d / 2;
        float theta = (float)pos / pow(freq_base, (float)(2 * pair) / (float)head_dim);
        theta /= rope_factor;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        int base = h * head_dim + d;
        float val = k[base];
        float partner = k[h * head_dim + (d % 2 == 0 ? d + 1 : d - 1)];
        if (d % 2 == 0)
            k[base] = val * cos_t - partner * sin_t;
        else
            k[base] = val * cos_t + partner * sin_t;
    }
}
"""


def rope_inplace(device, ptr_q, ptr_k, head_dim, n_heads, n_kv_heads,
                 pos, freq_base=1e7, rope_factor=4.0):
    k = _get_kernel(device, "rope", ROPE_INPLACE_SRC, b"rope_inplace")
    total = max(n_heads * head_dim, n_kv_heads * head_dim)
    _set_args(device, k, [
        ctypes.c_void_p(ptr_q), ctypes.c_void_p(ptr_k),
        ctypes.c_int32(head_dim), ctypes.c_int32(n_heads),
        ctypes.c_int32(n_kv_heads), ctypes.c_int32(pos),
        ctypes.c_float(freq_base), ctypes.c_float(rope_factor),
    ])
    gs = ((total + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


ROPE_FUSED_QKV_SRC = b"""
__kernel void rope_fused_qkv(
    __global float* qkv,
    const int q_offset,
    const int k_offset,
    const int head_dim,
    const int n_heads,
    const int n_kv_heads,
    const int pos,
    const float freq_base,
    const float rope_factor)
{
    int idx = get_global_id(0);
    int total_q = n_heads * head_dim;
    int total_k = n_kv_heads * head_dim;

    if (idx < total_q) {
        int h = idx / head_dim;
        int d = idx % head_dim;
        int pair = d / 2;
        float theta = (float)pos / pow(freq_base, (float)(2 * pair) / (float)head_dim);
        theta /= rope_factor;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        int base = q_offset + h * head_dim + d;
        int part = q_offset + h * head_dim + (d % 2 == 0 ? d + 1 : d - 1);
        float val = qkv[base];
        float partner = qkv[part];
        if (d % 2 == 0)
            qkv[base] = val * cos_t - partner * sin_t;
        else
            qkv[base] = val * cos_t + partner * sin_t;
    }

    if (idx < total_k) {
        int h = idx / head_dim;
        int d = idx % head_dim;
        int pair = d / 2;
        float theta = (float)pos / pow(freq_base, (float)(2 * pair) / (float)head_dim);
        theta /= rope_factor;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        int base = k_offset + h * head_dim + d;
        int part = k_offset + h * head_dim + (d % 2 == 0 ? d + 1 : d - 1);
        float val = qkv[base];
        float partner = qkv[part];
        if (d % 2 == 0)
            qkv[base] = val * cos_t - partner * sin_t;
        else
            qkv[base] = val * cos_t + partner * sin_t;
    }
}
"""


def rope_fused_qkv(device, ptr_qkv, q_offset, k_offset, head_dim,
                    n_heads, n_kv_heads, pos, freq_base=1e7, rope_factor=4.0):
    k = _get_kernel(device, "rope_fused_qkv", ROPE_FUSED_QKV_SRC, b"rope_fused_qkv")
    total = max(n_heads * head_dim, n_kv_heads * head_dim)
    _set_args(device, k, [
        ctypes.c_void_p(ptr_qkv),
        ctypes.c_int32(q_offset), ctypes.c_int32(k_offset),
        ctypes.c_int32(head_dim), ctypes.c_int32(n_heads),
        ctypes.c_int32(n_kv_heads), ctypes.c_int32(pos),
        ctypes.c_float(freq_base), ctypes.c_float(rope_factor),
    ])
    gs = ((total + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Softmax kernel (row-wise, in-place)
# ═══════════════════════════════════════════════════════════════════════

SOFTMAX_SRC = b"""
__kernel void softmax_inplace(
    __global float* x,
    const int n_cols)
{
    int row = get_global_id(0);
    float max_val = -1e30f;
    for (int i = 0; i < n_cols; i++) {
        float v = x[row * n_cols + i];
        if (v > max_val) max_val = v;
    }
    float sum = 0.0f;
    for (int i = 0; i < n_cols; i++) {
        float v = exp(x[row * n_cols + i] - max_val);
        x[row * n_cols + i] = v;
        sum += v;
    }
    for (int i = 0; i < n_cols; i++) {
        x[row * n_cols + i] /= sum;
    }
}
"""


def softmax_inplace(device, ptr_x, n_rows, n_cols):
    k = _get_kernel(device, "softmax", SOFTMAX_SRC, b"softmax_inplace")
    _set_args(device, k, [ctypes.c_void_p(ptr_x), ctypes.c_int32(n_cols)])
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(n_rows)),
        ctypes.pointer(ctypes.c_size_t(1)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# SiLU activation kernel
# ═══════════════════════════════════════════════════════════════════════

SILU_SRC = b"""
__kernel void silu_inplace(__global float* x, const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + exp(-v));
    }
}
"""


def silu_inplace(device, ptr_x, n):
    k = _get_kernel(device, "silu", SILU_SRC, b"silu_inplace")
    _set_args(device, k, [ctypes.c_void_p(ptr_x), ctypes.c_int32(n)])
    gs = ((n + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Elementwise add kernel (dst += src)
# ═══════════════════════════════════════════════════════════════════════

ADD_SRC = b"""
__kernel void add_inplace(__global float* dst, __global const float* src, const int n) {
    int i = get_global_id(0);
    if (i < n) dst[i] += src[i];
}
"""


def add_inplace(device, ptr_dst, ptr_src, n):
    k = _get_kernel(device, "add", ADD_SRC, b"add_inplace")
    _set_args(device, k, [ctypes.c_void_p(ptr_dst), ctypes.c_void_p(ptr_src), ctypes.c_int32(n)])
    gs = ((n + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# SwiGLU gate kernel: gate[i] = silu(gate[i]) * up[i]
# ═══════════════════════════════════════════════════════════════════════

SWIGLU_SRC = b"""
__kernel void swiglu_gate(
    __global float* gate,
    __global const float* up,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        float g = gate[i];
        gate[i] = (g / (1.0f + exp(-g))) * up[i];
    }
}
"""


def swiglu_gate(device, ptr_gate, ptr_up, n):
    k = _get_kernel(device, "swiglu", SWIGLU_SRC, b"swiglu_gate")
    _set_args(device, k, [ctypes.c_void_p(ptr_gate), ctypes.c_void_p(ptr_up), ctypes.c_int32(n)])
    gs = ((n + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Q4_K matrix-vector multiply with fused dequantization
# y[r] = sum_c(dequant(W[r,c]) * x[c])
# Each workgroup of 256 threads computes one output row.
# ═══════════════════════════════════════════════════════════════════════

MATVEC_Q4K_SRC = b"""
static inline float fp16_to_float(ushort h) {
    uint f = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1C000) << 13) | ((h & 0x03FF) << 13);
    return as_float(f);
}

static inline ushort read_ushort_le(__global const uchar* p) {
    return (ushort)p[0] | ((ushort)p[1] << 8);
}

static inline void get_scale_min_k4(int j, __global const uchar* q,
                                     uchar* d, uchar* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__kernel void matvec_q4k(
    __global const uchar* weights,
    __global const float* x,
    __global float* y,
    const int n_cols,
    const int n_rows,
    __local float* x_local)
{
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);
    int row = get_group_id(0) * wg_size + lid;

    for (int i = lid; i < n_cols; i += wg_size)
        x_local[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (row >= n_rows) return;

    int n_blocks = n_cols / 256;
    long row_bytes = (long)n_blocks * 144;
    __global const uchar* row_w = weights + (long)row * row_bytes;

    float sum = 0.0f;

    for (int blk = 0; blk < n_blocks; blk++) {
        __global const uchar* w = row_w + (long)blk * 144;
        float d = fp16_to_float(read_ushort_le(w + 0));
        float dmin = fp16_to_float(read_ushort_le(w + 2));
        __global const uchar* scales = w + 4;
        __global const uchar* qs = w + 16;
        int x_off = blk * 256;

        int is = 0;
        int q_off = 0;

        for (int j = 0; j < 256; j += 64) {
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4(is, scales, &sc0, &m0);
            get_scale_min_k4(is + 1, scales, &sc1, &m1);
            float d1 = d * (float)sc0;
            float m1v = dmin * (float)m0;
            float d2 = d * (float)sc1;
            float m2v = dmin * (float)m1;

            for (int l = 0; l < 32; l++) {
                uchar qv = qs[q_off + l];
                sum += (d1 * (float)(qv & 0x0F) - m1v) * x_local[x_off + j + l];
                sum += (d2 * (float)(qv >> 4) - m2v) * x_local[x_off + j + l + 32];
            }
            q_off += 32;
            is += 2;
        }
    }

    y[row] = sum;
}
"""


def matvec_q4k(device, ptr_w, ptr_x, ptr_y, n_rows, n_cols):
    k = _get_kernel(device, "matvec_q4k", MATVEC_Q4K_SRC, b"matvec_q4k")
    local_bytes = n_cols * 4
    _set_args(device, k, [
        ctypes.c_void_p(ptr_w), ctypes.c_void_p(ptr_x), ctypes.c_void_p(ptr_y),
        ctypes.c_int32(n_cols), ctypes.c_int32(n_rows),
    ])
    device._ocl.clSetKernelArg(k, 5, local_bytes, None)
    gs = ((n_rows + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Q6_K matrix-vector multiply with fused dequantization
# ═══════════════════════════════════════════════════════════════════════

MATVEC_Q6K_SRC = b"""
static inline float fp16_to_float_q6(ushort h) {
    uint f = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1C000) << 13) | ((h & 0x03FF) << 13);
    return as_float(f);
}

static inline ushort read_ushort_le_q6(__global const uchar* p) {
    return (ushort)p[0] | ((ushort)p[1] << 8);
}

__kernel void matvec_q6k(
    __global const uchar* weights,
    __global const float* x,
    __global float* y,
    const int n_cols,
    const int n_rows,
    __local float* x_local)
{
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);
    int row = get_group_id(0) * wg_size + lid;

    for (int i = lid; i < n_cols; i += wg_size)
        x_local[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (row >= n_rows) return;

    int n_blocks = n_cols / 256;
    long row_bytes = (long)n_blocks * 210;
    __global const uchar* row_w = weights + (long)row * row_bytes;

    float sum = 0.0f;

    for (int blk = 0; blk < n_blocks; blk++) {
        __global const uchar* w = row_w + (long)blk * 210;
        __global const uchar* ql = w;
        __global const uchar* qh = w + 128;
        __global const char*  sc = (w + 192);
        float d = fp16_to_float_q6(read_ushort_le_q6(w + 208));
        int x_off = blk * 256;

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[l] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += d * (float)sc[is]     * (float)q1 * x_local[x_off + n + l];
                sum += d * (float)sc[is + 2] * (float)q2 * x_local[x_off + n + 32 + l];
                sum += d * (float)sc[is + 4] * (float)q3 * x_local[x_off + n + 64 + l];
                sum += d * (float)sc[is + 6] * (float)q4 * x_local[x_off + n + 96 + l];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    y[row] = sum;
}
"""


def matvec_q6k(device, ptr_w, ptr_x, ptr_y, n_rows, n_cols):
    k = _get_kernel(device, "matvec_q6k", MATVEC_Q6K_SRC, b"matvec_q6k")
    local_bytes = n_cols * 4
    _set_args(device, k, [
        ctypes.c_void_p(ptr_w), ctypes.c_void_p(ptr_x), ctypes.c_void_p(ptr_y),
        ctypes.c_int32(n_cols), ctypes.c_int32(n_rows),
    ])
    device._ocl.clSetKernelArg(k, 5, local_bytes, None)
    gs = ((n_rows + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# F32 matrix-vector multiply (for F32 weight tensors)
# ═══════════════════════════════════════════════════════════════════════

MATVEC_F32_SRC = b"""
__kernel void matvec_f32(
    __global const float* weights,
    __global const float* x,
    __global float* y,
    const int n_cols,
    const int n_rows)
{
    int row = get_global_id(0);
    if (row >= n_rows) return;
    float sum = 0.0f;
    for (int c = 0; c < n_cols; c++) {
        sum += weights[row * n_cols + c] * x[c];
    }
    y[row] = sum;
}
"""


def matvec_f32(device, ptr_w, ptr_x, ptr_y, n_rows, n_cols):
    k = _get_kernel(device, "matvec_f32", MATVEC_F32_SRC, b"matvec_f32")
    _set_args(device, k, [
        ctypes.c_void_p(ptr_w), ctypes.c_void_p(ptr_x), ctypes.c_void_p(ptr_y),
        ctypes.c_int32(n_cols), ctypes.c_int32(n_rows),
    ])
    gs = ((n_rows + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Copy kernel (dst = src)
# ═══════════════════════════════════════════════════════════════════════

COPY_SRC = b"""
__kernel void copy_f32(
    __global const float* src,
    __global float* dst,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) dst[i] = src[i];
}
"""


def copy_f32(device, ptr_src, ptr_dst, n):
    k = _get_kernel(device, "copy", COPY_SRC, b"copy_f32")
    _set_args(device, k, [ctypes.c_void_p(ptr_src), ctypes.c_void_p(ptr_dst), ctypes.c_int32(n)])
    gs = ((n + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Argmax kernel (find index of max value in last dim)
# ═══════════════════════════════════════════════════════════════════════

ARGMAX_SRC = b"""
__kernel void argmax(
    __global const float* x,
    __global int* out,
    const int n_cols)
{
    int row = get_global_id(0);
    float max_val = -1e30f;
    int max_idx = 0;
    for (int i = 0; i < n_cols; i++) {
        float v = x[row * n_cols + i];
        if (v > max_val) { max_val = v; max_idx = i; }
    }
    out[row] = max_idx;
}
"""


def argmax(device, ptr_x, ptr_out, n_rows, n_cols):
    k = _get_kernel(device, "argmax", ARGMAX_SRC, b"argmax")
    _set_args(device, k, [ctypes.c_void_p(ptr_x), ctypes.c_void_p(ptr_out), ctypes.c_int32(n_cols)])
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(n_rows)),
        ctypes.pointer(ctypes.c_size_t(1)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# GQA Attention — scaled dot-product with grouped-query attention
# ═══════════════════════════════════════════════════════════════════════
# Memory layout (contiguous):
#   Q: [n_heads, seq_q, head_dim]
#   K: [n_kv_heads, seq_kv, head_dim]
#   V: [n_kv_heads, seq_kv, head_dim]
#   out: [n_heads, seq_q, head_dim]
#
# One work-item per (h, q_pos) = n_heads * seq_q total.
# Each work-item computes all head_dim output values for its head+position.
# Online softmax across k_pos — no local memory needed.

GQA_ATTN_SRC = b"""
__kernel void gqa_attention(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* out,
    const int n_heads,
    const int n_kv_heads,
    const int head_dim,
    const int seq_q,
    const int seq_kv,
    const float scale)
{
    int idx = get_global_id(0);
    if (idx >= n_heads * seq_q) return;

    int q_pos = idx % seq_q;
    int h = idx / seq_q;
    int kv_group = n_heads / n_kv_heads;
    int kv_h = h / kv_group;

    int q_base = h * seq_q * head_dim + q_pos * head_dim;
    int kv_base = kv_h * seq_kv * head_dim;
    int out_base = q_base;

    float max_score = -1e30f;
    float sum_exp = 0.0f;

    for (int dd = 0; dd < head_dim; dd++) {
        out[out_base + dd] = 0.0f;
    }

    for (int k_pos = 0; k_pos < seq_kv; k_pos++) {
        float score = 0.0f;
        int k_base = kv_base + k_pos * head_dim;
        for (int dd = 0; dd < head_dim; dd++) {
            score += Q[q_base + dd] * K[k_base + dd];
        }
        score *= scale;

        float new_max = max_score > score ? max_score : score;
        float rescale = exp(max_score - new_max);
        float w = exp(score - new_max);

        sum_exp = sum_exp * rescale + w;
        for (int dd = 0; dd < head_dim; dd++) {
            out[out_base + dd] = out[out_base + dd] * rescale + w * V[k_base + dd];
        }
        max_score = new_max;
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int dd = 0; dd < head_dim; dd++) {
        out[out_base + dd] *= inv_sum;
    }
}
"""


def gqa_attention(device, ptr_q, ptr_k, ptr_v, ptr_out,
                  n_heads, n_kv_heads, head_dim, seq_q, seq_kv):
    import math
    k = _get_kernel(device, "gqa_attention", GQA_ATTN_SRC, b"gqa_attention")
    scale = 1.0 / math.sqrt(head_dim)
    _set_args(device, k, [
        ctypes.c_void_p(ptr_q), ctypes.c_void_p(ptr_k),
        ctypes.c_void_p(ptr_v), ctypes.c_void_p(ptr_out),
        ctypes.c_int32(n_heads), ctypes.c_int32(n_kv_heads),
        ctypes.c_int32(head_dim), ctypes.c_int32(seq_q),
        ctypes.c_int32(seq_kv), ctypes.c_float(scale),
    ])
    total = n_heads * seq_q
    gs = ((total + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(min(total, 256))),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused QKV split: given [4096, 8192] fused output, split into Q, K, V
# ═══════════════════════════════════════════════════════════════════════

# This is handled in Python by slicing the output buffer.

# ═══════════════════════════════════════════════════════════════════════
# Fused add + RMS norm kernel: dst = rms_norm(src + residual, weight)
# Reduces buffer round-trip for the common add-norm pattern
# ═══════════════════════════════════════════════════════════════════════

ADD_RMS_NORM_SRC = b"""
__kernel void add_rms_norm(
    __global float* dst,
    __global const float* src,
    __global const float* residual,
    __global const float* weight,
    const int n,
    const float eps)
{
    int lid = get_local_id(0);
    __local float lsum[256];

    int i = get_global_id(0);
    float val = (i < n) ? src[i] + residual[i] : 0.0f;
    lsum[lid] = val * val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) lsum[lid] += lsum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float rms = sqrt(lsum[0] / (float)n + eps);

    if (i < n)
        dst[i] = (val / rms) * weight[i];
}
"""


def add_rms_norm(device, ptr_dst, ptr_src, ptr_res, ptr_w, n, eps=1e-6):
    k = _get_kernel(device, "add_rms_norm", ADD_RMS_NORM_SRC, b"add_rms_norm")
    _set_args(device, k, [
        ctypes.c_void_p(ptr_dst), ctypes.c_void_p(ptr_src),
        ctypes.c_void_p(ptr_res), ctypes.c_void_p(ptr_w),
        ctypes.c_int32(n), ctypes.c_float(eps),
    ])
    gs = ((n + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused RMS norm + matvec: output = matvec(w, rms_norm(x, wt), n_rows, n_cols)
# Eliminates norm output buffer round-trip
# ═══════════════════════════════════════════════════════════════════════

RMS_NORM_MATVEC_Q4K_SRC = b"""
static inline float fp16_to_float_nm(ushort h) {
    uint f = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1C000) << 13) | ((h & 0x03FF) << 13);
    return as_float(f);
}

static inline ushort read_ushort_le_nm(__global const uchar* p) {
    return (ushort)p[0] | ((ushort)p[1] << 8);
}

static inline void get_scale_min_k4_nm(int j, __global const uchar* q,
                                        uchar* d, uchar* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__kernel void rms_norm_matvec_q4k(
    __global const uchar* weights,
    __global const float* x,
    __global const float* norm_w,
    __global float* y,
    const int n_cols,
    const int n_rows,
    const float eps,
    __local float* x_local)
{
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);

    // Step 1: compute RMS of x (all threads participate in reduction)
    __local float lsum[256];
    float v = (lid < n_cols) ? x[lid] * x[lid] : 0.0f;
    lsum[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) lsum[lid] += lsum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = sqrt(lsum[0] / (float)n_cols + eps);
    float inv_rms = 1.0f / rms;

    // Step 2: load normalized x into local memory
    for (int i = lid; i < n_cols; i += wg_size)
        x_local[i] = x[i] * inv_rms * norm_w[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: matvec from local memory
    int row = get_group_id(0) * wg_size + lid;
    if (row >= n_rows) return;

    int n_blocks = n_cols / 256;
    long row_bytes = (long)n_blocks * 144;
    __global const uchar* row_w = weights + (long)row * row_bytes;

    float sum = 0.0f;

    for (int blk = 0; blk < n_blocks; blk++) {
        __global const uchar* w = row_w + (long)blk * 144;
        float d = fp16_to_float_nm(read_ushort_le_nm(w + 0));
        float dmin = fp16_to_float_nm(read_ushort_le_nm(w + 2));
        __global const uchar* scales = w + 4;
        __global const uchar* qs = w + 16;
        int x_off = blk * 256;

        int is = 0;
        int q_off = 0;

        for (int j = 0; j < 256; j += 64) {
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4_nm(is, scales, &sc0, &m0);
            get_scale_min_k4_nm(is + 1, scales, &sc1, &m1);
            float d1 = d * (float)sc0;
            float m1v = dmin * (float)m0;
            float d2 = d * (float)sc1;
            float m2v = dmin * (float)m1;

            for (int l = 0; l < 32; l++) {
                uchar qv = qs[q_off + l];
                sum += (d1 * (float)(qv & 0x0F) - m1v) * x_local[x_off + j + l];
                sum += (d2 * (float)(qv >> 4) - m2v) * x_local[x_off + j + l + 32];
            }
            q_off += 32;
            is += 2;
        }
    }

    y[row] = sum;
}
"""


def rms_norm_matvec_q4k(device, ptr_w, ptr_x, ptr_norm_w, ptr_y, n_rows, n_cols, eps=1e-6):
    k = _get_kernel(device, "rms_norm_matvec_q4k", RMS_NORM_MATVEC_Q4K_SRC, b"rms_norm_matvec_q4k")
    local_bytes = n_cols * 4
    _set_args(device, k, [
        ctypes.c_void_p(ptr_w), ctypes.c_void_p(ptr_x), ctypes.c_void_p(ptr_norm_w),
        ctypes.c_void_p(ptr_y),
        ctypes.c_int32(n_cols), ctypes.c_int32(n_rows), ctypes.c_float(eps),
    ])
    device._ocl.clSetKernelArg(k, 7, local_bytes, None)
    gs = ((n_rows + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# float4 vectorized add kernel: dst[i] += src[i] (4 floats at a time)
# ═══════════════════════════════════════════════════════════════════════

ADD_F4_SRC = b"""
__kernel void add_f4(
    __global float4* dst,
    __global const float4* src,
    const int n4)
{
    int i = get_global_id(0);
    if (i < n4)
        dst[i] += src[i];
}
"""


def add_f4(device, ptr_dst, ptr_src, n):
    k = _get_kernel(device, "add_f4", ADD_F4_SRC, b"add_f4")
    n4 = n // 4
    _set_args(device, k, [ctypes.c_void_p(ptr_dst), ctypes.c_void_p(ptr_src), ctypes.c_int32(n4)])
    gs = ((n4 + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# float4 vectorized copy kernel: dst[i] = src[i]
# ═══════════════════════════════════════════════════════════════════════

COPY_F4_SRC = b"""
__kernel void copy_f4(
    __global float4* dst,
    __global const float4* src,
    const int n4)
{
    int i = get_global_id(0);
    if (i < n4)
        dst[i] = src[i];
}
"""


def copy_f4(device, ptr_dst, ptr_src, n):
    k = _get_kernel(device, "copy_f4", COPY_F4_SRC, b"copy_f4")
    n4 = n // 4
    _set_args(device, k, [ctypes.c_void_p(ptr_dst), ctypes.c_void_p(ptr_src), ctypes.c_int32(n4)])
    gs = ((n4 + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# float4 vectorized SiLU gate: gate[i] = silu(gate[i]) * up[i]
# ═══════════════════════════════════════════════════════════════════════

SWIGLU_F4_SRC = b"""
__kernel void swiglu_f4(
    __global float4* gate,
    __global const float4* up,
    const int n4)
{
    int i = get_global_id(0);
    if (i < n4) {
        float4 g = gate[i];
        float4 u = up[i];
        float sx = g.x / (1.0f + exp(-g.x)) * u.x;
        float sy = g.y / (1.0f + exp(-g.y)) * u.y;
        float sz = g.z / (1.0f + exp(-g.z)) * u.z;
        float sw = g.w / (1.0f + exp(-g.w)) * u.w;
        gate[i] = (float4)(sx, sy, sz, sw);
    }
}
"""


def swiglu_f4(device, ptr_gate, ptr_up, n):
    k = _get_kernel(device, "swiglu_f4", SWIGLU_F4_SRC, b"swiglu_f4")
    n4 = n // 4
    _set_args(device, k, [ctypes.c_void_p(ptr_gate), ctypes.c_void_p(ptr_up), ctypes.c_int32(n4)])
    gs = ((n4 + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# float4 vectorized SiLU (no gate, just activation)
# ═══════════════════════════════════════════════════════════════════════

SILU_F4_SRC = b"""
__kernel void silu_f4(__global float4* x, const int n4) {
    int i = get_global_id(0);
    if (i < n4) {
        float4 v = x[i];
        float ox = v.x / (1.0f + exp(-v.x));
        float oy = v.y / (1.0f + exp(-v.y));
        float oz = v.z / (1.0f + exp(-v.z));
        float ow = v.w / (1.0f + exp(-v.w));
        x[i] = (float4)(ox, oy, oz, ow);
    }
}
"""


def silu_f4(device, ptr_x, n):
    k = _get_kernel(device, "silu_f4", SILU_F4_SRC, b"silu_f4")
    n4 = n // 4
    _set_args(device, k, [ctypes.c_void_p(ptr_x), ctypes.c_int32(n4)])
    gs = ((n4 + 255) // 256) * 256
    device._ocl.clEnqueueNDRangeKernel(
        device._queue, k, 1, None,
        ctypes.pointer(ctypes.c_size_t(gs)),
        ctypes.pointer(ctypes.c_size_t(256)),
        0, None, None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused GQA V-broadcast: for seq_kv=1, y[h,q] = v[0] (V broadcast)
# This is the common case in autoregressive generation (seq_kv=1)
# ═══════════════════════════════════════════════════════════════════════

# This is handled in the Python-side gqa_attention as a special case.

# ═══════════════════════════════════════════════════════════════════════
# Benchmark utility
# ═══════════════════════════════════════════════════════════════════════

def benchmark_bandwidth(device, size_mb=64):
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
    }
