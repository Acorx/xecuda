/**
 * @file xecuda_flash_attn.h
 * @brief FlashAttention-2 Acceleration for Intel Arc Xe2 XMX Matrix Engines
 */

#ifndef XECUDA_FLASH_ATTN_H
#define XECUDA_FLASH_ATTN_H

#include "xecuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief High-performance FlashAttention-2 for Intel Arc 130V GPU
 * Computes Output = Softmax(Q * K^T / sqrt(head_dim)) * V with online softmax tiling
 * 
 * @param Q Query tensor [batch_size, num_heads, seq_len, head_dim]
 * @param K Key tensor   [batch_size, num_heads, seq_len, head_dim]
 * @param V Value tensor [batch_size, num_heads, seq_len, head_dim]
 * @param O Output tensor[batch_size, num_heads, seq_len, head_dim]
 */
xeCudaError_t xeCudaFlashAttentionV2(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batchSize,
    int numHeads,
    int seqLen,
    int headDim
);

#ifdef __cplusplus
}
#endif

#endif // XECUDA_FLASH_ATTN_H
