"""
XeCUDA Model — Hybrid Mamba+Attention Forward Pass
===================================================
Runs the full Qwythos-9B forward pass entirely on the Intel Arc 130V GPU.
Keeps quantized weights on GPU and dequantizes fused inside matvec kernels.
Only activation tensors (hidden states) are F32 on GPU.

GGUF weight convention: shapes are [shape[0], shape[1]] in metadata but
actual data layout is [shape[1], shape[0]] (reversed). So for matvec:
  n_rows = shape[1]  (output dimension)
  n_cols = shape[0]  (input dimension)
"""

import ctypes
import numpy as np
from . import kernels as K

D_MODEL = 4096
D_FEEDFORWARD = 12288
N_HEADS = 16
N_KV_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 256
D_SSM = 32

BLOCK_IS_HYBRID = [
    1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0,
    1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0,
]


class Model:
    def __init__(self, device, weight_loader, ctx_len=2048):
        self.device = device
        self.wl = weight_loader

        f32_1 = lambda n: device.malloc(n * 4)
        self.buf_a = f32_1(D_MODEL)
        self.buf_b = f32_1(D_MODEL)
        self.buf_c = f32_1(D_MODEL)
        self.buf_d = f32_1(D_MODEL)
        self.buf_qkv = f32_1(8192)
        self.buf_k = f32_1(4096)
        self.buf_v = f32_1(4096)
        self.buf_gate_out = f32_1(D_MODEL)
        self.buf_ffn_gate = f32_1(D_FEEDFORWARD)
        self.buf_ffn_up = f32_1(D_FEEDFORWARD)
        self.buf_ssm_x = f32_1(D_MODEL)
        self.buf_ssm_out = f32_1(D_MODEL)
        self.buf_emb_row = f32_1(D_MODEL)
        self.buf_inv_rms = f32_1(1)
        self.buf_logits = None

    def _w(self, name):
        return self.wl.get(name)

    def _shape(self, name):
        return self.wl.info(name)["shape"]

    def _matvec(self, name, ptr_x, ptr_y, n_rows, n_cols):
        qt = self.wl.qtype(name)
        if qt == 14:
            K.matvec_q6k(self.device, self._w(name), ptr_x, ptr_y, n_rows, n_cols)
        elif qt == 12:
            K.matvec_q4k(self.device, self._w(name), ptr_x, ptr_y, n_rows, n_cols)
        else:
            K.matvec_f32(self.device, self._w(name), ptr_x, ptr_y, n_rows, n_cols)

    def _norm_matvec(self, norm_name, matvec_name, ptr_x, ptr_y, n_rows, n_cols):
        K.rms_norm_reduce(self.device, ptr_x, self.buf_inv_rms, n_cols)
        qt = self.wl.qtype(matvec_name)
        if qt == 12:
            K.norm_matvec_q4k(self.device, self._w(matvec_name), ptr_x,
                              self._w(norm_name), self.buf_inv_rms, ptr_y, n_rows, n_cols)
        else:
            K.rms_norm(self.device, ptr_x, self.buf_b, self._w(norm_name), 1, n_cols)
            self._matvec(matvec_name, self.buf_b, ptr_y, n_rows, n_cols)

    def _add_norm_matvec(self, norm_name, matvec_name, ptr_x, ptr_res, ptr_y, n_rows, n_cols):
        K.add_rms_norm_reduce(self.device, ptr_x, ptr_res, self.buf_inv_rms, n_cols)
        qt = self.wl.qtype(matvec_name)
        if qt == 12:
            K.add_norm_matvec_q4k(self.device, self._w(matvec_name), ptr_x, ptr_res,
                                  self._w(norm_name), self.buf_inv_rms, ptr_y, n_rows, n_cols)
        else:
            K.add_rms_norm(self.device, ptr_x, ptr_x, ptr_res, self._w(norm_name), n_cols)
            self._matvec(matvec_name, ptr_x, ptr_y, n_rows, n_cols)

    def _add_norm(self, norm_name, ptr_x, ptr_res, ptr_y, n):
        K.add_rms_norm_reduce(self.device, ptr_x, ptr_res, self.buf_inv_rms, n)
        K.add_norm(self.device, ptr_x, ptr_res, self._w(norm_name), self.buf_inv_rms, ptr_y, n)

    def forward(self, token_id, seq_pos=0):
        self._lookup_embedding(token_id, self.buf_emb_row)
        K.copy_f32(self.device, self.buf_emb_row, self.buf_a, D_MODEL)

        for layer_idx in range(32):
            is_hybrid = BLOCK_IS_HYBRID[layer_idx]
            if is_hybrid:
                self._forward_hybrid(layer_idx, seq_pos)
            else:
                self._forward_attn_only(layer_idx, seq_pos)

        self._norm_matvec("output_norm.weight", "output.weight",
                          self.buf_a, self.buf_b,
                          self._shape("output.weight")[1], D_MODEL)

        out_shape = self._shape("output.weight")
        vocab_size = out_shape[1]

        if self.buf_logits is None or self._logits_size != vocab_size:
            if self.buf_logits is not None:
                self.device.free(self.buf_logits)
            self.buf_logits = self.device.malloc(vocab_size * 4)
            self._logits_size = vocab_size
        self._matvec("output.weight", self.buf_b, self.buf_logits, vocab_size, out_shape[0])

        logits = np.empty(vocab_size, dtype=np.float32)
        self.device.read_buffer(self.buf_logits, logits.ctypes, vocab_size * 4)
        return logits

    def _forward_hybrid(self, li, seq_pos):
        s = self._shape

        # SSM: fused norm + ssm_out projection
        sh = s(f"blk.{li}.ssm_out.weight")
        self._norm_matvec(f"blk.{li}.post_attention_norm.weight",
                          f"blk.{li}.ssm_out.weight",
                          self.buf_a, self.buf_ssm_out, sh[1], sh[0])
        K.add_inplace(self.device, self.buf_a, self.buf_ssm_out, D_MODEL)

        # Attention: fused norm + QKV projection
        sh = s(f"blk.{li}.attn_qkv.weight")
        self._norm_matvec(f"blk.{li}.attn_norm.weight",
                          f"blk.{li}.attn_qkv.weight",
                          self.buf_a, self.buf_qkv, sh[1], sh[0])
        K.rope_fused_qkv(self.device, self.buf_qkv, 0, 4096,
                         HEAD_DIM, N_HEADS, N_KV_HEADS, seq_pos)
        sh = s(f"blk.{li}.attn_gate.weight")
        self._matvec(f"blk.{li}.attn_gate.weight",
                     self.buf_qkv, self.buf_gate_out, sh[1], sh[0])
        K.add_inplace(self.device, self.buf_a, self.buf_gate_out, D_MODEL)

        # FFN: norm → gate/up → swiglu → down → add
        # (norm feeds 2 matvecs, keep separate)
        K.rms_norm(self.device, self.buf_a, self.buf_b,
                   self._w(f"blk.{li}.post_attention_norm.weight"), 1, D_MODEL)
        sh = s(f"blk.{li}.ffn_gate.weight")
        self._matvec(f"blk.{li}.ffn_gate.weight",
                     self.buf_b, self.buf_ffn_gate, sh[1], sh[0])
        sh = s(f"blk.{li}.ffn_up.weight")
        self._matvec(f"blk.{li}.ffn_up.weight",
                     self.buf_b, self.buf_ffn_up, sh[1], sh[0])
        K.swiglu_f4(self.device, self.buf_ffn_gate, self.buf_ffn_up, D_FEEDFORWARD)
        sh = s(f"blk.{li}.ffn_down.weight")
        self._matvec(f"blk.{li}.ffn_down.weight",
                     self.buf_ffn_gate, self.buf_c, sh[1], sh[0])
        K.add_inplace(self.device, self.buf_a, self.buf_c, D_MODEL)

    def _forward_attn_only(self, li, seq_pos):
        s = self._shape

        K.rms_norm(self.device, self.buf_a, self.buf_b,
                   self._w(f"blk.{li}.attn_norm.weight"), 1, D_MODEL)

        sh = s(f"blk.{li}.attn_q.weight")
        self._matvec(f"blk.{li}.attn_q.weight",
                     self.buf_b, self.buf_qkv, sh[1], sh[0])
        sh = s(f"blk.{li}.attn_k.weight")
        self._matvec(f"blk.{li}.attn_k.weight",
                     self.buf_b, self.buf_k, sh[1], sh[0])
        sh = s(f"blk.{li}.attn_v.weight")
        self._matvec(f"blk.{li}.attn_v.weight",
                     self.buf_b, self.buf_v, sh[1], sh[0])

        K.rope_inplace(self.device, self.buf_qkv, self.buf_k,
                       HEAD_DIM, N_HEADS, N_KV_HEADS, seq_pos)

        K.rms_norm(self.device, self.buf_qkv, self.buf_qkv,
                   self._w(f"blk.{li}.attn_q_norm.weight"),
                   1, N_HEADS * HEAD_DIM)
        K.rms_norm(self.device, self.buf_k, self.buf_k,
                   self._w(f"blk.{li}.attn_k_norm.weight"),
                   1, N_KV_HEADS * HEAD_DIM)

        K.gqa_attention(self.device, self.buf_qkv, self.buf_k, self.buf_v,
                        self.buf_gate_out,
                        N_HEADS, N_KV_HEADS, HEAD_DIM, 1, 1)

        sh = s(f"blk.{li}.attn_output.weight")
        self._matvec(f"blk.{li}.attn_output.weight",
                     self.buf_gate_out, self.buf_c, sh[1], sh[0])
        K.add_inplace(self.device, self.buf_a, self.buf_c, D_MODEL)

        K.rms_norm(self.device, self.buf_a, self.buf_b,
                   self._w(f"blk.{li}.post_attention_norm.weight"), 1, D_MODEL)
        sh = s(f"blk.{li}.ffn_gate.weight")
        self._matvec(f"blk.{li}.ffn_gate.weight",
                     self.buf_b, self.buf_ffn_gate, sh[1], sh[0])
        sh = s(f"blk.{li}.ffn_up.weight")
        self._matvec(f"blk.{li}.ffn_up.weight",
                     self.buf_b, self.buf_ffn_up, sh[1], sh[0])
        K.swiglu_f4(self.device, self.buf_ffn_gate, self.buf_ffn_up, D_FEEDFORWARD)
        sh = s(f"blk.{li}.ffn_down.weight")
        self._matvec(f"blk.{li}.ffn_down.weight",
                     self.buf_ffn_gate, self.buf_c, sh[1], sh[0])
        K.add_inplace(self.device, self.buf_a, self.buf_c, D_MODEL)

    def _lookup_embedding(self, token_id, emb_out_buf):
        reader = self.wl.reader
        for t in reader.tensors:
            if t.name != "token_embd.weight":
                continue
            shape = [int(s) for s in t.shape]
            # Data layout is [shape[1], shape[0]] = [vocab, dim]
            # Token at offset token_id * dim
            dim = shape[0]
            vocab = shape[1]
            block_size = 256
            bytes_per_block = 144  # Q4_K

            # Each token's embedding is dim values = dim/256 blocks
            n_blocks_per_token = dim // block_size
            byte_offset = token_id * n_blocks_per_token * bytes_per_block
            row_bytes = n_blocks_per_token * bytes_per_block

            raw_data = t.data.tobytes()
            row_raw = raw_data[byte_offset:byte_offset + row_bytes]

            row_f32 = np.zeros(dim, dtype=np.float32)

            def _gsm4(j, sc):
                if j < 4:
                    return sc[j] & 63, sc[j + 4] & 63
                d = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)
                m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)
                return d, m

            for b in range(n_blocks_per_token):
                blk = row_raw[b * bytes_per_block:(b + 1) * bytes_per_block]
                d = np.frombuffer(blk[0:2], dtype=np.float16)[0].astype(np.float32)
                dmin = np.frombuffer(blk[2:4], dtype=np.float16)[0].astype(np.float32)
                scales = blk[4:16]
                qs = blk[16:144]

                is_idx = 0
                q_off = 0
                for j in range(0, 256, 64):
                    sc0, m0 = _gsm4(is_idx, scales)
                    sc1, m1 = _gsm4(is_idx + 1, scales)
                    d1 = d * float(sc0)
                    mv1 = dmin * float(m0)
                    d2 = d * float(sc1)
                    mv2 = dmin * float(m1)
                    for l in range(32):
                        qv = qs[q_off + l]
                        row_f32[b * 256 + j + l] = d1 * float(qv & 0x0F) - mv1
                        row_f32[b * 256 + j + l + 32] = d2 * float(qv >> 4) - mv2
                    q_off += 32
                    is_idx += 2
            break

        self.device.write_buffer(emb_out_buf, row_f32.ctypes, dim * 4)

    def shutdown(self):
        for attr in ['buf_a', 'buf_b', 'buf_c', 'buf_d',
                     'buf_qkv', 'buf_k', 'buf_v', 'buf_gate_out',
                     'buf_ffn_gate', 'buf_ffn_up', 'buf_ssm_x',
                     'buf_ssm_out', 'buf_emb_row']:
            buf = getattr(self, attr, None)
            if buf is not None:
                try:
                    self.device.free(buf)
                except Exception:
                    pass
