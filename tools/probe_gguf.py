import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from xecuda.gguf_reader import GGUFReader

m = r"C:/Users/arthu/.lmstudio/models/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"
r = GGUFReader(m)
print("version", r.version, "n_tensors", r.n_tensors, "size_GB %.2f" % (r.size / 1e9))
print("arch :", r.get("general.architecture"))
print("name :", r.get("general.name"))
print("layers:", r.get("qwen2.block_count"))
print("heads:", r.get("qwen2.attention.head_count"), "kv:", r.get("qwen2.attention.head_count_kv"))
print("hidden:", r.get("qwen2.embedding_length"), "ctx_len:", r.get("qwen2.context_length"))
special = [t for t in r.tensors if t.name in ("token_embd.weight", "output.weight")]
for t in special:
    print("  S", t.name, "gtype", t.gtype, "shape", t.shape)
blk0 = [t for t in r.tensors if "blk.0." in t.name]
print("blk.0 tensors (%d):" % len(blk0))
for t in blk0:
    print("   ", t.name, "gtype=%02d" % t.gtype, "shape", t.shape)
total = sum(r.tbytes(t) for t in r.tensors)
print("sum(tensor_bytes)=%.3f GB  file=%.3f GB" % (total / 1e9, r.size / 1e9))
rtype_ids = {}
for t in r.tensors:
    rtype_ids[t.gtype] = rtype_ids.get(t.gtype, 0) + 1
print("gtype histogram:", rtype_ids)
r.close()
print("OK")