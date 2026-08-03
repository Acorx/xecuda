"""GPU matvec on REAL GGUF quantized bytes vs llama.cpp official dequant.
Validates matvec_q4k (token_embd.weight Q4_K) and matvec_q6k (attn_qkv.weight Q6_K),
proving the fused decode reproduces llama.cpp on actual model weights."""
import os, sys, random, ctypes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctypes import c_float
from xecuda.device import XeCudaDevice
from xecuda import kernels as K
from xecuda.gguf_reader import GGUFReader
from _offref import dequant_q4k, dequant_q6_K

MODEL = r"C:/Users/arthu/.lmstudio/models/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"

def check_row(d, kern, tensor_name, width, blk_bytes, blk_elems, dequant, r):
    t = [t for t in r.tensors if t.name == tensor_name][0]
    raw = r.read_raw(t)
    rb = (width // blk_elems) * blk_bytes
    rng = random.Random(7)
    x = [rng.uniform(-1, 1) for _ in range(width)]
    xbuf = (c_float * width)(*x)
    px = d.malloc(width * 4); d.write_buffer(px, ctypes.byref(xbuf), width * 4)
    maxdiff = 0.0
    for rr in (0, 3, 9):
        wslice = raw[rr * rb:(rr + 1) * rb]
        ref = dequant(wslice, width // blk_elems)
        yref = sum(a * b for a, b in zip(ref, x))
        pw = d.malloc(len(wslice)); d.write_buffer(pw, wslice, len(wslice))
        py = d.malloc(4)
        kern(d, pw, px, py, 1, width)
        o = (c_float * 1)(); d.read_buffer(py, ctypes.byref(o), 4); d.finish()
        maxdiff = max(maxdiff, abs(o[0] - yref))
    return maxdiff

def main():
    d = XeCudaDevice()
    r = GGUFReader(MODEL)
    d4 = check_row(d, K.matvec_q4k, "token_embd.weight", 4096, 144, 256, dequant_q4k, r)
    d6 = check_row(d, K.matvec_q6k, "blk.0.attn_qkv.weight", 4096, 210, 256, dequant_q6_K, r)
    print("Q4_K token_embd : maxdiff=%.3e  %s" % (d4, "PASS" if d4 < 1e-3 else "FAIL"))
    print("Q6_K attn_qkv   : maxdiff=%.3e  %s" % (d6, "PASS" if d6 < 1e-3 else "FAIL"))
    r.close(); d.shutdown()

if __name__ == "__main__":
    main()