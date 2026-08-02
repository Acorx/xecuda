"""
XeCUDA — real decoder-block forward on the GPU (OpenCL, no / numpy / exe)
==========================================================================
One transformer decoder block (RMSNorm -> QKV -> GQA attention -> output proj
-> SwiGLU MLP -> residuals) run with the project's real OpenCL kernels on the
Intel Arc, checked against a pure-Python mirror on the same weights.
"""
import os, sys, math, time, ctypes, random
from ctypes import c_float

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from xecuda.device import XeCudaDevice
from xecuda import kernels as K
from validate_real_gpu import ref_rms_norm, ref_gqa

D = 32; NH, NKV, HD = 4, 2, 8; SQ, SK = 3, 4

def h2b(d, f):
    n = len(f); a = (c_float * n)(*f); h = d.malloc(n * 4)
    d.write_buffer(h, ctypes.byref(a), n * 4); return h
def uv(d, h, n):
    a = (c_float * n)(); d.read_buffer(h, ctypes.byref(a), n * 4); d.finish(); return [float(v) for v in a]
def gmv(d, W, x):
    R = len(W) // len(x); o = d.malloc(R * 4)
    K.matvec_f32(d, h2b(d, W), h2b(d, x), o, R, len(x)); d.finish(); r = uv(d, o, R); d.free(o); return r
def grms(d, x, w):
    o = d.malloc(len(x) * 4)
    K.rms_norm(d, h2b(d, x), o, h2b(d, w), 1, len(x)); d.finish(); r = uv(d, o, len(x)); d.free(o); return r
def gsilu(d, x):
    o = h2b(d, x); K.silu_inplace(d, o, len(x)); d.finish(); r = uv(d, o, len(x)); d.free(o); return r
def fanout(seqvecs, ngroup, seqlen):
    out = []
    for g in range(ngroup):
        for t in range(seqlen):
            for v in range(HD):
                out.append(seqvecs[t][g * HD + v])
    return out
def gattn(d, Qv, Kv, Vv):
    Qf, Kf, Vf = fanout(Qv, NH, SQ), fanout(Kv, NKV, SK), fanout(Vv, NKV, SK)
    o = d.malloc(NH * SQ * HD * 4)
    K.gqa_attention(d, h2b(d, Qf), h2b(d, Kf), h2b(d, Vf), o, NH, NKV, HD, SQ, SK)
    a = uv(d, o, NH * SQ * HD); d.free(o); return a

def gpu_block(d, Wq, Wk, Wv, Wo, Wup, Wgat, Wdo, w2, X):
    t0 = time.perf_counter()
    Qv = [gmv(d, Wq, X[t]) for t in range(SQ)]
    Kv = [gmv(d, Wk, X[t]) for t in range(SK)]
    Vv = [gmv(d, Wv, X[t]) for t in range(SK)]
    attn = gattn(d, Qv, Kv, Vv)
    out = []
    for t in range(SQ):
        a_t = [attn[h * SQ * HD + t * HD + v] for h in range(NH) for v in range(HD)]
        proj = gmv(d, Wo, a_t)
        x_r = [X[t][i] + proj[i] for i in range(D)]
        rn = grms(d, x_r, w2)
        u  = gmv(d, Wup, rn); g = gmv(d, Wgat, rn)
        act = [gsilu(d, [u[i]])[0] * g[i] for i in range(4 * D)]
        down = gmv(d, Wdo, act)
        out.append([x_r[i] + down[i] for i in range(D)])
    return out, time.perf_counter() - t0

def ref_block(Wq, Wk, Wv, Wo, Wup, Wgat, Wdo, w2, X):
    def mve(W, x): return [sum(W[r * len(x) + i] * x[i] for i in range(len(x))) for r in range(len(W) // len(x))]
    def fan(seq, n, slen): return [seq[t][g * HD + v] for g in range(n) for t in range(slen) for v in range(HD)]
    Qv = [mve(Wq, X[t]) for t in range(SQ)]; Kv = [mve(Wk, X[t]) for t in range(SK)]; Vv = [mve(Wv, X[t]) for t in range(SK)]
    attn = ref_gqa(fan(Qv, NH), fan(Kv, NKV), fan(Vv, NKV), NH, NKV, HD, SQ, SK, 1 / math.sqrt(D * 1)) if False else ref_gqa(fan(Qv, NH, SQ), fan(Kv, NKV, SK), fan(Vv, NKV, SK), NH, NKV, HD, SQ, SK, 1 / math.sqrt(HD))
    out = []
    for t in range(SQ):
        a_t = [attn[h * SQ * HD + t * HD + v] for h in range(NH) for v in range(HD)]
        proj = mve(Wo, a_t); x_r = [X[t][i] + proj[i] for i in range(D)]
        r   = ref_rms_norm(x_r, w2)
        ua  = mve(Wup, r); gc  = mve(Wgat, r)
        act = [(ua[i] / (1 + math.exp(-ua[i]))) * gc[i] for i in range(4 * D)]
        dn  = mve(Wdo, act)
        out.append([x_r[i] + dn[i] for i in range(D)])
    return out

if __name__ == "__main__":
    print("XeCUDA decoder-block forward on the real GPU (OpenCL)")
    print("-" * 52)
    dev = XeCudaDevice()
    try:
        rg = random.Random(7); M = lambda r_, c_: [rg.random() * 2 - 1 for _ in range(r_ * c_)]
        Wq = M(NH * HD, D); Wk = M(NKV * HD, D); Wv = M(NKV * HD, D); Wo = M(D, NH * HD)
        Wup = M(4 * D, D); Wgat = M(4 * D, D); Wdo = M(D, 4 * D)
        w2 = [1.0 + (i % 5) * 0.02 for i in range(D)]
        X  = [[rg.random() * 2 - 1 for _ in range(D)] for _ in range(SK)]
        g, gt = gpu_block(dev, Wq, Wk, Wv, Wo, Wup, Wgat, Wdo, w2, X)
        rr = ref_block(Wq, Wk, Wv, Wo, Wup, Wgat, Wdo, w2, X)
        e = max(abs(g[t][i] - rr[t][i]) for t in range(SQ) for i in range(D))
        print("  wall-clock (GPU)  : %.3f ms" % (gt * 1000))
        print("  max |GPU - ref|    : %.3e" % e)
        print("  -> %s" % ("PASS (GPU block == reference)" if e < 1e-2 else "FAIL"))
    finally:
        dev.shutdown()