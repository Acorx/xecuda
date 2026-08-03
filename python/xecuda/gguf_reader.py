"""
GGUF reader — pure stdlib, for llama.cpp GGUF (v3+) files.
Parses header + metadata + tensor index; raw quantized bytes read lazily by
offset. GGML layouts mapped so a Q4_K_M model (tensors of GGML_TYPE_Q4_K)
can feed the matvec_q4k device kernel directly.
"""
from __future__ import annotations
import os, struct
from typing import List, NamedTuple

# GGUF value types
U8, I8, U16, I16, U32, I32, F32V = 0, 1, 2, 3, 4, 5, 6
BOOL, STRING, ARRAY, U64, I64, F64V = 7, 8, 9, 10, 11, 12

# GGML tensor types
FT32, FT16 = 0, 1
QT40, QT41, QT50, QT51, QT80, QT81 = 2, 3, 4, 5, 6, 7
QK2, QK3, QK4, QK5, QK6, QK8 = 10, 11, 12, 13, 14, 15

# (block_bytes, block_elems)
_GGML = {FT32: (4, 1), FT16: (2, 1),
         QT40: (18, 32), QT41: (20, 32), QT50: (22, 32), QT51: (24, 32),
         QT80: (34, 32), QT81: (36, 32),
         QK2: (128, 256), QK3: (112, 256), QK4: (144, 256),
         QK5: (176, 256), QK6: (210, 256), QK8: (240, 256)}


def ggml_size(gtype: int, n_elems: int) -> int:
    if gtype == FT32: return n_elems * 4
    if gtype == FT16: return n_elems * 2
    bb, be = _GGML[gtype]
    return ((n_elems + be - 1) // be) * bb


class TensorInfo(NamedTuple):
    name: str
    gtype: int
    shape: List[int]  # logical dims
    offset: int


class _R:
    def __init__(self, f): self.f = f
    def _u(self, fmt, n):
        b = self.f.read(n)
        if len(b) < n: raise EOFError("GGUF truncated")
        return struct.unpack(fmt, b)[0]
    def u8(self):  return self._u("B", 1)
    def i8(self):  return self._u("b", 1)
    def u16(self): return self._u("H", 2)
    def i16(self): return self._u("h", 2)
    def u32(self): return self._u("I", 4)
    def i32(self): return self._u("i", 4)
    def u64(self): return self._u("Q", 8)
    def i64(self): return self._u("q", 8)
    def f32(self): return self._u("f", 4)
    def f64(self): return self._u("d", 8)
    def blobs(self, n):
        b = self.f.read(n)
        if len(b) < n: raise EOFError("GGUF truncated")
        return b
    def string(self): return self.blobs(self.u64()).decode("utf-8")


class GGUFReader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.r = _R(self.f)
        if self.r.blobs(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        self.version = self.r.u32()
        self.n_tensors = self.r.u64()
        self.n_kv = self.r.u64()
        self.metadata = {}
        for _ in range(self.n_kv):
            k = self.r.string()
            self.metadata[k] = self._value(self.r.u32())
        self.tensors: List[TensorInfo] = []

        for _ in range(self.n_tensors):
            name = self.r.string()
            nd = self.r.u32()
            shape = [self.r.u64() for _ in range(nd)][::-1]
            gt = self.r.u32()
            off = self.r.u64()
            self.tensors.append(TensorInfo(name, gt, shape, off))
        self.size = os.path.getsize(path)
        self._finalize_base()

    def _prim(self, vt):
        if vt == U8:    return self.r.u8()
        if vt == I8:    return self.r.i8()
        if vt == U16:   return self.r.u16()
        if vt == I16:   return self.r.i16()
        if vt == U32:   return self.r.u32()
        if vt == I32:   return self.r.i32()
        if vt == F32V:  return self.r.f32()
        if vt == U64:   return self.r.u64()
        if vt == I64:   return self.r.i64()
        if vt == F64V:  return self.r.f64()
        if vt == BOOL:  return self.r.u8() != 0
        if vt == STRING: return self.r.string()
        raise ValueError("unknown GGUF value type %d" % vt)

    def _value(self, vt):
        if vt == ARRAY:
            et = self.r.u32()
            n = self.r.u64()
            if et == ARRAY:
                return [self._value(et) for _ in range(n)]
            return [self._prim(et) for _ in range(n)]
        return self._prim(vt)

    # ── helpers ────────────────────────────────────────────────────────
    def _finalize_base(self):
        a = self.metadata.get("general.alignment", 32)
        base = self.f.tell()
        self.f.seek(base + (a - base % a) % a)
        self.data_base = self.f.tell()

    def get(self, key, default=None):
        return self.metadata.get(key, default)
    def tensors_by(self, prefix=""):
        return [t for t in self.tensors if t.name.startswith(prefix)]
    def nelems(self, t):
        n = 1
        for x in t.shape: n *= x
        return n
    def tbytes(self, t):
        return ggml_size(t.gtype, self.nelems(t))
    def read_raw(self, t):
        self.f.seek(self.data_base + t.offset)
        return self.f.read(self.tbytes(t))
    def close(self):
        self.f.close()