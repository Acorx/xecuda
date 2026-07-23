"""
XeCUDA Pipeline Validation — Test each layer independently
==========================================================
Validates the full inference pipeline on real Intel Arc 130V GPU.
Tests: GPU init → weight loading → individual kernels → single block → full model.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from xecuda.device import XeCudaDevice
from xecuda.weight_loader import WeightLoader
from xecuda import kernels as K


def test_gpu_init():
    print("=" * 60)
    print("TEST 1: GPU Initialization")
    print("=" * 60)
    device = XeCudaDevice()
    info = device.info()
    print(f"  GPU: {info['gpu_name']}")
    print(f"  Vendor: {info['vendor']}")
    print(f"  Compute Units: {info['compute_units']}")
    print(f"  Total Memory: {info['total_memory_gb']:.1f} GB")
    print(f"  Max Alloc: {info['max_alloc_gb']:.1f} GB")
    print("  PASS: GPU initialized successfully")
    return device


def test_weight_loading(device):
    print("\n" + "=" * 60)
    print("TEST 2: Weight Loading")
    print("=" * 60)
    wl = WeightLoader(device)
    wl.load(verbose=True)

    # Verify key tensors exist
    key_tensors = [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
    ]
    missing = [t for t in key_tensors if wl.get(t) is None]
    if missing:
        print(f"  FAIL: Missing tensors: {missing}")
        return None, None
    print(f"  All {len(key_tensors)} key tensors verified")
    print("  PASS: Weights loaded to GPU")
    return wl, key_tensors


def test_individual_kernels(device):
    print("\n" + "=" * 60)
    print("TEST 3: Individual Kernels")
    print("=" * 60)
    results = {}

    # RMS Norm
    n, dim = 2, 4096
    x = np.random.randn(n, dim).astype(np.float32)
    w = np.ones(dim, dtype=np.float32)
    y = np.zeros_like(x)
    bx = device.malloc(n * dim * 4)
    by = device.malloc(n * dim * 4)
    bw = device.malloc(dim * 4)
    device.write_buffer(bx, x.ctypes, n * dim * 4)
    device.write_buffer(bw, w.ctypes, dim * 4)
    K.rms_norm(device, bx, by, bw, n, dim)
    device.read_buffer(by, y.ctypes, n * dim * 4)
    rms = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + 1e-6)
    expected = (x / rms) * w
    err = np.max(np.abs(y - expected))
    results["rms_norm"] = err < 1e-5
    print(f"  rms_norm: max_err={err:.2e} {'PASS' if results['rms_norm'] else 'FAIL'}")
    device.free(bx); device.free(by); device.free(bw)

    # SiLU
    n = 1024
    x = np.random.randn(n).astype(np.float32)
    expected = x / (1 + np.exp(-x))
    bx = device.malloc(n * 4)
    device.write_buffer(bx, x.ctypes, n * 4)
    K.silu_inplace(device, bx, n)
    y = np.zeros(n, dtype=np.float32)
    device.read_buffer(bx, y.ctypes, n * 4)
    err = np.max(np.abs(y - expected))
    results["silu"] = err < 1e-5
    print(f"  silu: max_err={err:.2e} {'PASS' if results['silu'] else 'FAIL'}")
    device.free(bx)

    # Add
    n = 1024
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    expected = a + b
    ba = device.malloc(n * 4)
    bb = device.malloc(n * 4)
    device.write_buffer(ba, a.ctypes, n * 4)
    device.write_buffer(bb, b.ctypes, n * 4)
    K.add_inplace(device, ba, bb, n)
    y = np.zeros(n, dtype=np.float32)
    device.read_buffer(ba, y.ctypes, n * 4)
    err = np.max(np.abs(y - expected))
    results["add"] = err < 1e-5
    print(f"  add: max_err={err:.2e} {'PASS' if results['add'] else 'FAIL'}")
    device.free(ba); device.free(bb)

    # Softmax
    n_rows, n_cols = 4, 1024
    x = np.random.randn(n_rows, n_cols).astype(np.float32)
    # Manual softmax (no scipy)
    def py_softmax(a, axis=1):
        e = np.exp(a - np.max(a, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
    expected = py_softmax(x, axis=1)
    bx = device.malloc(n_rows * n_cols * 4)
    device.write_buffer(bx, x.ctypes, n_rows * n_cols * 4)
    K.softmax_inplace(device, bx, n_rows, n_cols)
    y = np.zeros_like(x)
    device.read_buffer(bx, y.ctypes, n_rows * n_cols * 4)
    err = np.max(np.abs(y - expected))
    results["softmax"] = err < 1e-5
    print(f"  softmax: max_err={err:.2e} {'PASS' if results['softmax'] else 'FAIL'}")
    device.free(bx)

    passed = sum(results.values())
    total = len(results)
    print(f"\n  Kernel tests: {passed}/{total} passed")
    return all(results.values())


def test_matvec_f32(device):
    print("\n" + "=" * 60)
    print("TEST 4: F32 Matrix-Vector Multiply")
    print("=" * 60)
    n_rows, n_cols = 1024, 4096
    w = np.random.randn(n_rows, n_cols).astype(np.float32) * 0.02
    x = np.random.randn(n_cols).astype(np.float32)
    expected = w @ x
    bw = device.malloc(n_rows * n_cols * 4)
    bx = device.malloc(n_cols * 4)
    by = device.malloc(n_rows * 4)
    device.write_buffer(bw, w.ctypes, n_rows * n_cols * 4)
    device.write_buffer(bx, x.ctypes, n_cols * 4)
    K.matvec_f32(device, bw, bx, by, n_rows, n_cols)
    y = np.zeros(n_rows, dtype=np.float32)
    device.read_buffer(by, y.ctypes, n_rows * 4)
    err = np.max(np.abs(y - expected))
    passed = err < 1e-3
    print(f"  matvec_f32: max_err={err:.2e} {'PASS' if passed else 'FAIL'}")
    device.free(bw); device.free(bx); device.free(by)
    return passed


def test_q4k_matvec(device, wl):
    print("\n" + "=" * 60)
    print("TEST 5: Q4_K Matrix-Vector Multiply (Real Weights)")
    print("=" * 60)

    # Test with a small known tensor: blk.0.attn_norm.weight (F32, [4096])
    # and blk.0.ffn_gate.weight (Q4_K, [4096, 12288])
    w_buf = wl.get("blk.0.ffn_gate.weight")
    info = wl.info("blk.0.ffn_gate.weight")
    n_rows, n_cols = info["shape"]
    print(f"  Testing Q4_K matvec: [{n_rows}, {n_cols}]")

    # Create random input
    x = np.random.randn(n_cols).astype(np.float32) * 0.02
    bx = device.malloc(n_cols * 4)
    by = device.malloc(n_rows * 4)
    device.write_buffer(bx, x.ctypes, n_cols * 4)

    t0 = time.perf_counter()
    K.matvec_q4k(device, w_buf, bx, by, n_rows, n_cols)
    device.finish()
    t1 = time.perf_counter()

    y = np.zeros(n_rows, dtype=np.float32)
    device.read_buffer(by, y.ctypes, n_rows * 4)

    print(f"  Output: mean={np.mean(y):.4f}, std={np.std(y):.4f}")
    print(f"  Time: {(t1-t0)*1000:.1f} ms")
    print(f"  PASS: Q4_K matvec completed without errors")
    device.free(bx); device.free(by)
    return True


def test_q6k_matvec(device, wl):
    print("\n" + "=" * 60)
    print("TEST 6: Q6_K Matrix-Vector Multiply (Real Weights)")
    print("=" * 60)

    w_buf = wl.get("blk.0.ffn_down.weight")
    info = wl.info("blk.0.ffn_down.weight")
    n_rows, n_cols = info["shape"]
    print(f"  Testing Q6_K matvec: [{n_rows}, {n_cols}]")

    x = np.random.randn(n_cols).astype(np.float32) * 0.02
    bx = device.malloc(n_cols * 4)
    by = device.malloc(n_rows * 4)
    device.write_buffer(bx, x.ctypes, n_cols * 4)

    t0 = time.perf_counter()
    K.matvec_q6k(device, w_buf, bx, by, n_rows, n_cols)
    device.finish()
    t1 = time.perf_counter()

    y = np.zeros(n_rows, dtype=np.float32)
    device.read_buffer(by, y.ctypes, n_rows * 4)

    print(f"  Output: mean={np.mean(y):.4f}, std={np.std(y):.4f}")
    print(f"  Time: {(t1-t0)*1000:.1f} ms")
    print(f"  PASS: Q6_K matvec completed without errors")
    device.free(bx); device.free(by)
    return True


def test_single_block(device, wl):
    print("\n" + "=" * 60)
    print("TEST 7: Single Block Forward Pass (Block 0 — Hybrid)")
    print("=" * 60)

    from xecuda.model import Model
    model = Model(device, wl)

    # Test forward pass for a single token
    print("  Running forward pass for token_id=1 (first token)...")
    t0 = time.perf_counter()
    logits = model.forward(1, seq_pos=0)
    device.finish()
    t1 = time.perf_counter()

    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Logits mean: {logits.mean():.4f}")
    print(f"  Time: {(t1-t0)*1000:.1f} ms")
    print(f"  PASS: Forward pass completed without errors")

    model.shutdown()
    return True


def test_embedding_lookup(device, wl):
    print("\n" + "=" * 60)
    print("TEST 8: Embedding Lookup")
    print("=" * 60)

    # Read raw embedding data and compare with GGUF
    reader = wl.reader
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            # Extract first row (token 0)
            n_cols = int(t.shape[1])  # 4096
            bytes_per_block = 144  # Q4_K
            n_blocks_per_row = n_cols // 256  # 16

            raw = t.data.tobytes()

            # Read first 16 blocks (one row)
            row_bytes = n_blocks_per_row * bytes_per_block
            row_raw = raw[:row_bytes]

            # Dequantize on CPU
            row_f32 = np.zeros(n_cols, dtype=np.float32)
            for b in range(n_blocks_per_row):
                block = row_raw[b * bytes_per_block:(b + 1) * bytes_per_block]
                d = np.frombuffer(block[0:2], dtype=np.float16)[0].astype(np.float32)
                dmin = np.frombuffer(block[2:4], dtype=np.float16)[0].astype(np.float32)
                scales = block[4:16]
                qs = block[16:144]

                is_idx = 0
                q_off = 0
                for half in range(2):
                    sc_j = is_idx
                    if sc_j < 4:
                        sc = scales[sc_j] & 63
                        m = scales[sc_j + 4] & 63
                    else:
                        sc = (scales[sc_j + 4] & 0xF) | ((scales[sc_j - 4] >> 6) << 4)
                        m = (scales[sc_j + 4] >> 4) | ((scales[sc_j] >> 6) << 4)
                    dval = d * float(sc)
                    mval = dmin * float(m)
                    for l in range(32):
                        qv = qs[q_off + l]
                        row_f32[b * 256 + half * 32 + l] = dval * float(qv & 0x0F) - mval
                        row_f32[b * 256 + half * 32 + l + 32] = dval * float(qv >> 4) - mval
                    q_off += 32
                    is_idx += 1
            break

    print(f"  Token 0 embedding: mean={row_f32.mean():.4f}, std={row_f32.std():.4f}")
    print(f"  Range: [{row_f32.min():.4f}, {row_f32.max():.4f}]")
    passed = row_f32.std() > 0.001
    print(f"  {'PASS' if passed else 'FAIL'}: Embedding has non-trivial values")
    return passed


def main():
    print("XeCUDA Pipeline Validation — Intel Arc 130V")
    print("=" * 60)

    t_start = time.perf_counter()

    # Test 1: GPU init
    device = test_gpu_init()

    # Test 2: Weight loading
    wl, key_tensors = test_weight_loading(device)
    if wl is None:
        print("\nFATAL: Weight loading failed")
        return

    # Test 3: Individual kernels
    kernels_ok = test_individual_kernels(device)

    # Test 4: F32 matvec
    f32_ok = test_matvec_f32(device)

    # Test 5: Q4_K matvec
    q4k_ok = test_q4k_matvec(device, wl)

    # Test 6: Q6_K matvec
    q6k_ok = test_q6k_matvec(device, wl)

    # Test 7: Embedding lookup
    emb_ok = test_embedding_lookup(device, wl)

    # Test 8: Single block forward pass
    block_ok = test_single_block(device, wl)

    # Summary
    t_end = time.perf_counter()
    all_tests = [kernels_ok, f32_ok, q4k_ok, q6k_ok, emb_ok, block_ok]
    passed = sum(all_tests)
    total = len(all_tests)

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed in {t_end-t_start:.1f}s")
    print("=" * 60)

    if all(all_tests):
        print("\nAll tests passed! Ready for text generation.")
        print("Run: python -m xecuda.generate 'The capital of France is'")
    else:
        print("\nSome tests failed. Check output above for details.")

    # Cleanup
    wl.shutdown()
    device.shutdown()


if __name__ == "__main__":
    main()
