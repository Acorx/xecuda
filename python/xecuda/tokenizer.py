r"""Byte-level BPE tokenizer built from a GGUF vocab (gpt2/qwen style).

Reads tokenizer.ggml.* metadata via GGUFReader and implements:
  - encode(text) -> list[int]   (byte-level BPE + qwen pre-tokenizer regex)
  - decode(ids)  -> str
Special tokens (token_type == 3) are kept verbatim on both paths.
Pure stdlib + `regex` (unicode \p{L}/\p{N} classes).
"""
import regex

# Qwen2.5/3 pre-tokenizer pattern (also used by gguf pre=qwen35)
QWEN_RE = regex.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

def bytes_to_unicode():
    """Standard GPT-2 byte -> unicode visible-char map (space -> 'Ġ' U+0120)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))

class BPETokenizer:
    def __init__(self, reader=None, tokens=None, merges=None, token_type=None):
        if reader is not None:
            tokens = reader.get("tokenizer.ggml.tokens")
            merges = reader.get("tokenizer.ggml.merges")
            token_type = reader.get("tokenizer.ggml.token_type")
        self.tokens = tokens
        self.token_type = token_type or [1] * len(tokens)
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.merge_ranks = {}
        for rank, m in enumerate(merges):
            a, b = m.split(" ", 1)
            self.merge_ranks[(a, b)] = rank
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.special_ids = [i for i, tt in enumerate(self.token_type) if tt == 3]
        self.special_tokens = [tokens[i] for i in self.special_ids]
        self.special_to_id = {tokens[i]: i for i in self.special_ids}
        self._bpe_cache = {}

    # ---- BPE core (canonical GPT-2 tuple algorithm, first-occurrence merge) ----
    def _get_pairs(self, word):
        pairs = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def _bpe(self, token):
        if token in self._bpe_cache:
            return self._bpe_cache[token]
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return (token,)
        while True:
            bigram = min(pairs, key=lambda p: self.merge_ranks.get(p, 1 << 30))
            if bigram not in self.merge_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            n = len(word)
            while i < n:
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < n - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        self._bpe_cache[token] = word
        return word

    # ---- public API ----
    def encode(self, text, add_special=None):
        if add_special is None:
            add_special = [t for t in ("<|im_start|>", "<|im_end|>") if t in self.special_to_id]
        ids = []
        # split on special tokens first (verbatim handling)
        if self.special_to_id:
            import re as _re
            pat = _re.compile("(" + "|".join(_re.escape(t) for t in self.special_to_id) + ")")
            parts = pat.split(text)
        else:
            parts = [text]
        for part in parts:
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
                continue
            raw = "".join(self.byte_encoder[b] for b in part.encode("utf-8"))
            for piece in QWEN_RE.findall(raw):
                ids.extend(self.token_to_id[t] for t in self._bpe(piece))
        return ids

    def decode(self, ids, skip_special=False):
        out = []
        for i in ids:
            if skip_special and i in set(self.special_ids):
                continue
            out.append(self.tokens[i])
        text = "".join(out)
        try:
            return bytes(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")
        except KeyError:  # a non-byte char slipped in (shouldn't happen)
            return text
