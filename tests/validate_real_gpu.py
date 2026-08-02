"""
XeCUDA — Real OpenCL GPU validation (dependency-free)
======================================================
Pure ctypes / OpenCL. No numpy, no torch, no gguf needed.
Each kernel is checked for CORRECTNESS against a pure-Python reference,
then a real wall-clock time is reported.

This is the honest "does the GPU actually compute correctly" suite.
"""
import os, sys, math, time, ctypes
from ctypes import c_float

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from xecuda.device import XeCudaDevice
from xecuda import kernels as K


# ────────────────────────────────────────────────────────────────────────
# helpers: move host floats <-> GPU cl_mem buffers
# ────────────────────────────────────────────────────────────────────────
def h2d(dev, floats):
    n = len(floats)
    arr = (c_float * n)(*floats)
    h = dev.malloc(n * 4)
    dev.write_buffer(h, ctypes.byref(arr), n * 4)
    return h

def d2h(dev, h, n):
    arr = (c_float * n)()
    dev.read_buffer(h, ctypes.byref(arr), n * 4)
    dev.finish()
    return list(arr)

def report(name, ok, extra=""):
    flag = "PASS" if ok else "FAIL"
    mark = "PASS" if ok else ">><<"
    print(f"  [{mark}] {name}  {extra}")
    return ok

# ────────────────────────────────────────────────────────────────────────
# pure-Python correctness references
# ────────────────────────────────────────────────────────────────────────
def matmul(A, B, M, Kk, N):
    return [[sum(A[r * Kk + c] * B[c * N + n] for c in range(Kk)) for n in range(N)] for r in range(M)]

def ref_rms_norm(x, w, eps=1e-6):
    s = sum(v * v for v in x) / len(x)
    r = 1.0 / math.sqrt(s + eps)
    return [xi * r * wi for xi, wi in zip(x, w)]

def ref_softmax_row(x, n_rows, n_cols):
    out = list(x)
    for r in range(n_rows):
        row = out[r*n_cols:(r+1)*n_cols]
        m = max(row); e = [math.exp(v - m) for v in row]; s = sum(e)
        out[r*n_cols:(r+1)*n_cols] = [v / s for v in e]
    return out

def ref_gqa(Q, Kth, Vval, n_heads, n_kv_heads, head_dim, seq_q, seq_kv, scale):
    if n_kv_heads == 0:
        n_kv_heads = n_heads
    kv_group = n_heads // n_kv_heads
    out = [0.0] * (n_heads * seq_q * head_dim)
    for h in range(n_heads):
        kvh = h // kv_group if kv_group > 0 else h
        qbase0 = h * seq_q * head_dim
        kbase0 = kvh * seq_kv * head_dim
        for qp in range(seq_q):
            acc = [0.0] * head_dim
            m = -1e30; s = 0.0
            qbase = qbase0 + qp * head_dim
            for kp in range(seq_kv):
                kbase = kbase0 + kp * head_dim
                sc = sum(Q[qbase + d] * Kth[kbase + d] for d in range(head_dim)) * scale
                nm = max(m, sc); w = math.exp(sc - nm); rs = math.exp(m - nm)
                s = s * rs + w
                vbase = kbase0 + kp * head_dim
                for d in range(head_dim):
                    acc[d] = acc[d] * rs + w * Vval[vbase + d]
                m = nm
            for d in range(head_dim):
                out[qbase + d] = acc[d] / s
    return out


# ────────────────────────────────────────────────────────────────────────
def run(dev):
    ok = True
    print("  Device :", dev.info())
    ok_all = True

    # 1. vector_add
    N = 1 << 12
    a = [float(i) for i in range(N)]
    b = [float(N - i) for i in range(N)]
    pa, pb, pc = h2d(dev, a), h2d(dev, b), dev.malloc(N * 4)
    K.vector_add(dev, pa, pb, pc, N); dev.finish()
    c = d2h(dev, pc, N)
    e = sum(1 for i in range(N) if abs(c[i] - (a[i] + b[i])) > 1e-3)
    ok &= report("vector_add 4096", e == 0, f"(err={e})")
    dev.free(pa); dev.free(pb); dev.free(pc)

    # 2. sgemm
    M, Kk, Nn = 32, 32, 32
    Aflat = [1.0 if r == c else 0.01 for r in range(M) for c in range(Kk)]
    Bflat = [2.0 for _ in range(Kk * Nn)]
    pA, pB, pC = h2d(dev, Aflat), h2d(dev, Bflat), dev.malloc(M * Nn * 4)
    t0 = time.perf_counter(); K.sgemm(dev, pA, pB, pC, M, Nn, Kk); dev.finish()
    ms = (time.perf_counter() - t0) * 1000
    C = d2h(dev, pC, M * Nn)
    ref = matmul(Aflat, Bflat, M, Kk, Nn)
    e = sum(1 for r in range(M) for cc in range(Nn) if abs(C[r*Nn+cc] - ref[r][cc]) > 1e-2)
    ok &= report("sgemm 32x32x32", e == 0, f"err={e}, {ms:.2f}ms")
    dev.free(pA); dev.free(pB); dev.free(pC)

    # 3. matvec_f32
    R, Cc = 16, 64
    W = [0.5 for _ in range(R * Cc)]; x = [float(i) for i in range(Cc)]
    y = [sum(W[r*Cc+i] * x[i] for i in range(Cc)) for r in range(R)]
    pW, px, py = h2d(dev, W), h2d(dev, x), dev.malloc(R * 4)
    K.matvec_f32(dev, pW, px, py, R, Cc); dev.finish()
    got = d2h(dev, py, R)
    e = sum(1 for i in range(R) if abs(got[i] - y[i]) > 0.01)
    ok &= report("matvec_f32", e == 0, f"err={e}")

    # 4. rms_norm  (kernel computes ONE RMS over the flat array == one row)
    nrows, dim = 1, 128
    xx = [float((i % 7) - 3) for i in range(nrows * dim)]
    ww = [1.0 + (i % 5) * 0.1 for i in range(dim)]
    pxx, pww, py = h2d(dev, xx), h2d(dev, ww), dev.malloc(nrows * dim * 4)
    K.rms_norm(dev, pxx, py, pww, nrows, dim); dev.finish()
    got = d2h(dev, py, nrows * dim)
    want = ref_rms_norm(xx[:dim], ww)
    e = sum(1 for i in range(dim) if abs(got[i] - want[i]) > 1e-2)
    ok &= report("rms_norm", e == 0, f"err={e}")

    # 5. softmax_inplace
    nr, nc = 4, 8
    ss = [float(i % 11 - 5) for i in range(nr * nc)]
    ps = h2d(dev, ss); K.softmax_inplace(dev, ps, nr, nc); dev.finish()
    got = d2h(dev, ps, nr * nc)
    want = ref_softmax_row(ss, nr, nc)
    e = sum(1 for i in range(len(ss)) if abs(got[i] - want[i]) > 1e-3)
    ok &= report("softmax_inplace", e == 0, f"err={e}")

    # 6. gqa_attention
    heads, kv, hdim, sq, sk = 1, 1, 8, 4, 6
    q = [float((i % 13) - 6) / 3 for i in range(heads * sq * hdim)]
    k_ = [float((i % 9) - 4) / 3 for i in range(kv * sk * hdim)]
    v_ = [float((i % 5) + 1) / 2 for i in range(kv * sk * hdim)]
    pq, pk, pv, po = h2d(dev, q), h2d(dev, k_), h2d(dev, v_), dev.malloc(heads * sq * hdim * 4)
    K.gqa_attention(dev, pq, pk, pv, po, heads, kv, hdim, sq, sk); dev.finish()
    got = d2h(dev, po, heads * sq * hdim)
    want = ref_gqa(q, k_, v_, heads, kv, hdim, sq, sk, 1.0 / math.sqrt(hdim))
    e = sum(1 for i in range(len(want)) if abs(got[i] - want[i]) > 1e-3)
    ok &= report("gqa_attention", e == 0, f"err={e}")

    # 7. silu_inplace
    n = 512
    vals = [float(i - 200) for i in range(n)]
    pxv = h2d(dev, vals); K.silu_inplace(dev, pxv, n); dev.finish()
    got = d2h(dev, pxv, n)
    want = [v / (1.0 + math.exp(-v)) for v in vals]
    e = sum(1 for i in range(n) if abs(got[i] - want[i]) > 1e-3)
    ok &= report("silu_inplace", e == 0, f"err={e}")
    dev.free(pxv)

    # 8. bandwidth (real timing)
    bw = K.benchmark_bandwidth(dev, size_mb=16)
    print(f"  └── bandwidth write={bw['write_bw_gbs']} GB/s read={bw['read_bw_gbs']} GB/s")

    return ok


if __name__ == "__main__":
    print("XeCUDA real-GPU validation (OpenCL, no numpy)")
    print("-" * 44)
    dev = XeCudaDevice()
    try:
        all_ok = run(dev)
        print("-" * 44)
        print("RESULT:", "ALL PASS ✔" if all_ok else "SOME FAIL ✖")
        sys.exit(0 if all_ok else 1)
    finally:
        dev.shutdown()