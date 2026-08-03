"""p3 verification: BPE tokenizer on the REAL Qwythos-9B GGUF vocab.
Checks: byte-encoder bijectivity, exact known-token encodings, and
decode(encode(text)) == text round-trips on French/English/code samples."""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
from xecuda.gguf_reader import GGUFReader
from xecuda.tokenizer import BPETokenizer

MODEL = r"C:/Users/arthu/.lmstudio/models/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"

t0 = time.time()
r = GGUFReader(MODEL)
tok = BPETokenizer(r)
print("tokenizer built in %.1fs  (vocab=%d, merges=%d, specials=%d)" %
      (time.time() - t0, len(tok.tokens), len(tok.merge_ranks), len(tok.special_ids)))

# 1) byte encoder bijectivity
enc = tok.byte_encoder
assert len(set(enc.values())) == 256 and len(enc) == 256, "byte encoder not bijective"
print("[1] byte encoder bijective (256/256)")

# 2) known single-token encodings
def ids(s): return tok.encode(s)
assert ids("Hello") == [tok.token_to_id["Hello"]], ids("Hello")
assert ids("Hello world") == [tok.token_to_id["Hello"], tok.token_to_id["Ġworld"]], ids("Hello world")
print("[2] known encodings OK: 'Hello'->1 token, 'Hello world'->[Hello][Ġworld]")

# 3) round-trips
samples = [
    "Bonjour le monde !",
    "L'été arrive très vite, n'est-ce pas ?",
    "Hello, world! This is a test of 12345 tokens.",
    "def add(a, b):\n    return a + b  # commentaire",
    "Café déjà-vu naïve piñata œuf — accents ok",
    "Les nombres: 42, 3.14159, -17, 1e10.",
    "Texte avec des mots composés français : aujourd'hui, peut-être.",
    "Mixed Émoji 🔥 test 🚀 and code `x := 1`",
    "",
    "  espaces   multiples\tet tabulations\nfins de ligne\r\nCRLF",
]
fails = 0
for s in samples:
    ids_ = tok.encode(s)
    back = tok.decode(ids_)
    ok = back == s
    fails += (not ok)
    print("[3] roundtrip %-70r -> %3d ids  %s" % (s[:68], len(ids_), "OK" if ok else "MISMATCH: %r" % back))
print("[3] round-trips: %d/%d pass" % (len(samples) - fails, len(samples)))

# 4) special tokens survive
s = "<|im_start|>system\nTu es utile.<|im_end|>\n<|im_start|>user\nBonjour<|im_end|>"
ids_ = tok.encode(s)
back = tok.decode(ids_)
print("[4] specials: ids=%d... roundtrip %s" % (len(ids_), "OK" if back == s else "MISMATCH: %r" % back))

# 5) token-count sanity: ~1.3 tokens per word typical
s = ("Le développement de l'intelligence artificielle sur les GPU Intel Arc nécessite "
     "une pile logicielle complète, des kernels OpenCL aux modèles quantifiés. " * 5)
ids_ = tok.encode(s)
print("[5] %d chars -> %d ids (ratio %.2f)" % (len(s), len(ids_), len(ids_) / max(len(s.split()), 1)))
r.close()
print("RESULT:", "PASS" if fails == 0 else "FAIL")