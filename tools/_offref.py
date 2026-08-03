"""Official llama.cpp quant dequant references (ported to pure Python).
dequantize_row_q4_K and dequantize_row_q6_K from ggml-quants.c.
Block layouts from ggml-common.h:
  Q4_K: d(2) dmin(2) scales(12) qs(128)        = 144 B / 256 elems
  Q6_K: ql(128) qh(64) scales(16 i8) d(2)      = 210 B / 256 elems
"""
import struct

def fp16f(h):
    s = (h >> 15) & 1; e = (h >> 10) & 0x1F; m = h & 0x3FF
    if e == 0:  return (-1.) ** s * (m / 1024.) * 2 ** -14
    if e == 31: return (-1.) ** s * (1.0 if m == 0 else float('nan'))
    return (-1.) ** s * (1 + m / 1024.) * 2 ** (e - 15)

def _get(b): return fp16f(struct.unpack('<H', b)[0])

def get_scale_min_k4(j, q):
    if j < 4: return q[j] & 63, q[j + 4] & 63
    return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4))

def dequant_q4k(row, nblk):
    """dequantize_row_q4_K: 32 low nibbles then 32 high per 64-group."""
    out = []
    for b in range(nblk):
        w = row[b * 144:(b + 1) * 144]
        d = _get(w[0:2]); mn = _get(w[2:4]); q = w[16:144]; sc_ = w[4:16]
        is_ = 0; qi = 0
        for j in range(0, 256, 64):
            s0, m0 = get_scale_min_k4(is_, sc_)
            s1, m1 = get_scale_min_k4(is_ + 1, sc_)
            d1 = d * s0; m1v = mn * m0; d2 = d * s1; m2v = mn * m1
            for l in range(32): out.append(d1 * (q[qi + l] & 0xF) - m1v)
            for l in range(32): out.append(d2 * (q[qi + l] >> 4) - m2v)
            qi += 32; is_ += 2
    return out

def dequant_q6_K(row, nb):
    """dequantize_row_q6_K: 16 blocks of 16 (sc 8-bit), ql/qh nibble-bits."""
    out = []
    for b in range(nb):
        w = row[b * 210:(b + 1) * 210]
        d = _get(w[208:210]); ql = w[0:128]; qh = w[128:192]
        sc = [x - 256 if x >= 128 else x for x in w[192:208]]
        vals = [0.0] * 256
        for nn in range(2):
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[nn*64+l]     & 0xF) | (((qh[nn*32+l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[nn*64+l+32]  & 0xF) | (((qh[nn*32+l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[nn*64+l]     >> 4)  | (((qh[nn*32+l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[nn*64+l+32]  >> 4)  | (((qh[nn*32+l] >> 6) & 3) << 4)) - 32
                base = nn * 128
                vals[base + l]       = d * sc[nn*8 + is_ + 0] * q1
                vals[base + l + 32]  = d * sc[nn*8 + is_ + 2] * q2
                vals[base + l + 64]  = d * sc[nn*8 + is_ + 4] * q3
                vals[base + l + 96]  = d * sc[nn*8 + is_ + 6] * q4
        out.extend(vals)
    return out