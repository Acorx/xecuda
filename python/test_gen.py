import sys, os, traceback, warnings, time
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

sys.path.insert(0, os.path.dirname(__file__))
try:
    from xecuda.device import XeCudaDevice
    from xecuda.weight_loader import WeightLoader
    from xecuda.model import Model, D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_IS_HYBRID
    from xecuda.generate import SimpleTokenizer
    import xecuda.kernels as K
    import numpy as np
    import ctypes, time

    device = XeCudaDevice()
    wl = WeightLoader(device)
    wl.load(verbose=False)
    model = Model(device, wl)
    tokenizer = SimpleTokenizer()
    tokenizer.load()

    token_id = tokenizer.encode('Hello')[0]
    print(f'Input token: {token_id}', flush=True)

    t0 = time.perf_counter()
    model._lookup_embedding(token_id, model.buf_emb_row)
    K.copy_f32(device, model.buf_emb_row, model.buf_a, D_MODEL)

    for layer_idx in range(32):
        tl0 = time.perf_counter()
        if BLOCK_IS_HYBRID[layer_idx]:
            model._forward_hybrid(layer_idx, 0)
        else:
            model._forward_attn_only(layer_idx, 0)
        device.finish()
        tl1 = time.perf_counter()
        label = 'HYB' if BLOCK_IS_HYBRID[layer_idx] else 'ATN'
        print(f'  blk.{layer_idx:2d} [{label}] {(tl1-tl0)*1000:.1f}ms', flush=True)

    K.rms_norm(device, model.buf_a, model.buf_b,
               model._w("output_norm.weight"), 1, D_MODEL)
    out_shape = model._shape("output.weight")
    vocab_size = out_shape[1]
    logits_buf = device.malloc(vocab_size * 4)
    model._matvec("output.weight", model.buf_b, logits_buf, vocab_size, out_shape[0])
    device.finish()
    t1 = time.perf_counter()

    logits = np.empty(vocab_size, dtype=np.float32)
    device.read_buffer(logits_buf, logits.ctypes, vocab_size * 4)
    device.free(logits_buf)

    top5 = np.argsort(logits)[-5:][::-1]
    print(f'\nTop-5 logits ({(t1-t0)*1000:.0f}ms total):', flush=True)
    for rank, tid in enumerate(top5):
        try:
            decoded = tokenizer.decode([int(tid)])
        except:
            decoded = f'<token-{tid}>'
        print(f'  {rank+1}. token {tid} logit={logits[tid]:.2f}', flush=True)

except Exception as e:
    traceback.print_exc()
