/**
 * @file xecuda_blas.h
 * @brief cuBLAS Equivalent & Native GGUF Quantized MatMul for Intel Arc XMX
 */

#ifndef XECUDA_BLAS_H
#define XECUDA_BLAS_H

#include "xecuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum xeCublasOperation {
    XE_CUBLAS_OP_N = 0, // No Transpose
    XE_CUBLAS_OP_T = 1, // Transpose
    XE_CUBLAS_OP_C = 2  // Conjugate Transpose
} xeCublasOperation_t;

typedef struct xeCublasContext* xeCublasHandle_t;

xeCudaError_t xeCublasCreate(xeCublasHandle_t* handle);
xeCudaError_t xeCublasDestroy(xeCublasHandle_t handle);

/**
 * @brief Single-precision General Matrix Multiply (GEMM) on Intel Arc GPU
 * Computes C = alpha * op(A) * op(B) + beta * C
 */
xeCudaError_t xeCublasSgemm(
    xeCublasHandle_t handle,
    xeCublasOperation_t transa,
    xeCublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc
);

/**
 * @brief Native XeCUDA Q4_K_M Block Dequantization & Matrix-Vector Multiplication
 * Computes y = W_q4 * x directly on Intel Arc Xe2 Vector/XMX Cores
 */
xeCudaError_t xeCudaMatVecQ4KM(
    const uint8_t* q4_weights,
    const float* x_in,
    float* y_out,
    int rows,
    int cols
);

#ifdef __cplusplus
}
#endif

#endif // XECUDA_BLAS_H
