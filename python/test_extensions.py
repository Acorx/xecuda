import ctypes, sys, numpy as np
sys.path.insert(0, '.')
from xecuda.device import XeCudaDevice

dev = XeCudaDevice()

# Test local memory
src = b"""
__kernel void test_local(__global const float* in, __global float* out, const int n) {
    __local float buf[256];
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    if (gid < n) buf[lid] = in[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (gid < n) out[gid] = buf[lid] * 2.0f;
}
"""
k = dev.build_kernel('test_local', src, b'test_local')
a = np.arange(256, dtype=np.float32)
a_buf = dev.malloc(256*4); dev.write_buffer(a_buf, a.ctypes, 256*4)
b_buf = dev.malloc(256*4)
dev.enqueue_kernel(k, 256, 256, [ctypes.c_void_p(a_buf), ctypes.c_void_p(b_buf), ctypes.c_int32(256)])
dev.finish()
b = np.empty(256, dtype=np.float32)
dev.read_buffer(b_buf, b.ctypes, 256*4)
ok = np.allclose(b, a * 2)
print("Local memory test:", "OK" if ok else "FAIL", "got", b[:4])

# Test larger local memory (16KB for x cache)
src_big = b"""
__kernel void test_local_big(__global const float* in, __global float* out, const int n) {
    __local float buf[4096];
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int wg_size = get_local_size(0);
    for (int i = lid; i < n; i += wg_size) buf[i] = in[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (gid < n) out[gid] = buf[gid] + 1.0f;
}
"""
k2 = dev.build_kernel('test_local_big', src_big, b'test_local_big')
a2 = np.ones(4096, dtype=np.float32) * 3.0
a2_buf = dev.malloc(4096*4); dev.write_buffer(a2_buf, a2.ctypes, 4096*4)
b2_buf = dev.malloc(4096*4)
dev.enqueue_kernel(k2, 4096, 256, [ctypes.c_void_p(a2_buf), ctypes.c_void_p(b2_buf), ctypes.c_int32(4096)])
dev.finish()
b2 = np.empty(4096, dtype=np.float32)
dev.read_buffer(b2_buf, b2.ctypes, 4096*4)
ok2 = np.allclose(b2, 4.0)
print("Local mem 4096:", "OK" if ok2 else "FAIL", "got", b2[:4])

# Benchmark: local vs global for matvec-like pattern
src_global = b"""
__kernel void bench_global(__global const float* x, __global const uchar* w, __global float* y,
                           const int n_cols, const int n_rows) {
    int row = get_global_id(0);
    if (row >= n_rows) return;
    float sum = 0.0f;
    for (int c = 0; c < n_cols; c++) sum += (float)w[row * n_cols + c] * x[c];
    y[row] = sum;
}
"""
src_local = b"""
__kernel void bench_local(__global const float* x, __global const uchar* w, __global float* y,
                          const int n_cols, const int n_rows) {
    __local float xbuf[4096];
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);
    int gid = get_global_id(0);
    for (int i = lid; i < n_cols; i += wg_size) xbuf[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (gid >= n_rows) return;
    float sum = 0.0f;
    for (int c = 0; c < n_cols; c++) sum += (float)w[gid * n_cols + c] * xbuf[c];
    y[gid] = sum;
}
"""
# Small test
n_rows, n_cols = 4096, 4096
x_np = np.random.randn(n_cols).astype(np.float32)
w_np = np.random.randint(0, 255, (n_rows, n_cols), dtype=np.uint8)
y_g = np.zeros(n_rows, dtype=np.float32)
y_l = np.zeros(n_rows, dtype=np.float32)
x_buf = dev.malloc(n_cols*4); dev.write_buffer(x_buf, x_np.ctypes, n_cols*4)
w_buf = dev.malloc(n_rows*n_cols); dev.write_buffer(w_buf, w_np.ctypes, n_rows*n_cols)
y_g_buf = dev.malloc(n_rows*4)
y_l_buf = dev.malloc(n_rows*4)

kg = dev.build_kernel('bench_global', src_global, b'bench_global')
_set_args_args = [ctypes.c_void_p(x_buf), ctypes.c_void_p(w_buf), ctypes.c_void_p(y_g_buf), ctypes.c_int32(n_cols), ctypes.c_int32(n_rows)]
dev.enqueue_kernel(kg, n_rows, 1, _set_args_args)
dev.finish()

kl = dev.build_kernel('bench_local', src_local, b'bench_local')
dev.enqueue_kernel(kl, n_rows, 256, [ctypes.c_void_p(x_buf), ctypes.c_void_p(w_buf), ctypes.c_void_p(y_l_buf), ctypes.c_int32(n_cols), ctypes.c_int32(n_rows)])
dev.finish()

dev.read_buffer(y_g_buf, y_g.ctypes, n_rows*4)
dev.read_buffer(y_l_buf, y_l.ctypes, n_rows*4)
ok3 = np.allclose(y_g, y_l, atol=1e-3)
print("Local vs global matvec:", "OK" if ok3 else "FAIL")

dev.free(x_buf); dev.free(w_buf); dev.free(y_g_buf); dev.free(y_l_buf)
dev.free(a_buf); dev.free(b_buf); dev.free(a2_buf); dev.free(b2_buf)
