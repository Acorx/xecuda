/**
 * @file xecuda_blas.cpp
 * @brief BLAS & Q4_K_M Matrix-Vector — Real GPU Dispatch via OpenCL
 */

#include "../include/xecuda_blas.h"
#include "../include/xecuda_ocl.h"
#include <iostream>
#include <cstring>
#include <mutex>

struct xeCublasContext {
    int activeDevice;
};

// ============================================================================
// SGEMM kernel source (compiled on GPU at first call)
// ============================================================================

static const char* SGEMM_SRC =
    "__kernel void sgemm_cl(\n"
    "    const int M, const int N, const int K,\n"
    "    __global const float* A, __global const float* B,\n"
    "    __global float* C,\n"
    "    const float alpha, const float beta)\n"
    "{\n"
    "    int row = get_global_id(0);\n"
    "    int col = get_global_id(1);\n"
    "    if (row < M && col < N) {\n"
    "        float sum = 0.0f;\n"
    "        for (int k = 0; k < K; k++)\n"
    "            sum += A[row * K + k] * B[k * N + col];\n"
    "        C[row * N + col] = alpha * sum + beta * C[row * N + col];\n"
    "    }\n"
    "}\n";

static const char* VEC_ADD_SRC =
    "__kernel void vec_add_cl(\n"
    "    __global const float* A, __global const float* B,\n"
    "    __global float* C, const int N)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i < N) C[i] = A[i] + B[i];\n"
    "}\n";

static const char* MATVEC_SRC =
    "__kernel void matvec_cl(\n"
    "    __global const uchar* q4, __global const float* x,\n"
    "    __global float* y, const int rows, const int cols)\n"
    "{\n"
    "    int r = get_global_id(0);\n"
    "    if (r >= rows) return;\n"
    "    int hc = cols / 2;\n"
    "    float sum = 0.0f;\n"
    "    for (int c = 0; c < hc; c++) {\n"
    "        uchar b = q4[r * hc + c];\n"
    "        float w0 = (float)((b & 0x0F) - 8) * 0.0625f;\n"
    "        float w1 = (float)(((b >> 4) & 0x0F) - 8) * 0.0625f;\n"
    "        int col = c * 2;\n"
    "        if (col < cols)     sum += w0 * x[col];\n"
    "        if (col + 1 < cols) sum += w1 * x[col + 1];\n"
    "    }\n"
    "    y[r] = sum;\n"
    "}\n";

static cl_program  g_sgemmProg  = nullptr;
static cl_kernel   g_sgemmKernel = nullptr;
static cl_program  g_matvecProg = nullptr;
static cl_kernel   g_matvecKernel = nullptr;
static std::once_flag g_sgemmFlag;
static std::once_flag g_matvecFlag;

static bool ensureSgemmKernel() {
    std::call_once(g_sgemmFlag, []() {
        if (!xoc::isInitialized()) return;
        g_sgemmProg = xoc::gpuCompileProgram(SGEMM_SRC, std::strlen(SGEMM_SRC));
        if (g_sgemmProg) g_sgemmKernel = xoc::gpuCreateKernel(g_sgemmProg, "sgemm_cl");
    });
    return g_sgemmKernel != nullptr;
}

static bool ensureMatvecKernel() {
    std::call_once(g_matvecFlag, []() {
        if (!xoc::isInitialized()) return;
        g_matvecProg = xoc::gpuCompileProgram(MATVEC_SRC, std::strlen(MATVEC_SRC));
        if (g_matvecProg) g_matvecKernel = xoc::gpuCreateKernel(g_matvecProg, "matvec_cl");
    });
    return g_matvecKernel != nullptr;
}

// ============================================================================
// Public API
// ============================================================================

xeCudaError_t xeCublasCreate(xeCublasHandle_t* handle) {
    if (!handle) return xeCudaErrorInvalidValue;
    xeCublasContext* ctx = new xeCublasContext{0};
    *handle = ctx;
    return xeCudaSuccess;
}

xeCudaError_t xeCublasDestroy(xeCublasHandle_t handle) {
    if (handle) delete handle;
    return xeCudaSuccess;
}

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
) {
    if (!handle || !A || !B || !C || !alpha || !beta) return xeCudaErrorInvalidValue;
    if (transa != XE_CUBLAS_OP_N || transb != XE_CUBLAS_OP_N) return xeCudaErrorNotYetImplemented;
    if (!xoc::isInitialized()) return xeCudaErrorInitializationError;
    if (!ensureSgemmKernel()) return xeCudaErrorInitializationError;

    // A, B, C are device pointers (cl_mem handles) per CUDA convention
    cl_mem dA = (cl_mem)A;
    cl_mem dB = (cl_mem)B;
    cl_mem dC = (cl_mem)C;

    cl_int mM = m, nN = n, kK = k;
    xoc::gpuSetArg(g_sgemmKernel, 0, sizeof(cl_int), &mM);
    xoc::gpuSetArg(g_sgemmKernel, 1, sizeof(cl_int), &nN);
    xoc::gpuSetArg(g_sgemmKernel, 2, sizeof(cl_int), &kK);
    xoc::gpuSetArg(g_sgemmKernel, 3, sizeof(cl_mem), &dA);
    xoc::gpuSetArg(g_sgemmKernel, 4, sizeof(cl_mem), &dB);
    xoc::gpuSetArg(g_sgemmKernel, 5, sizeof(cl_mem), &dC);
    xoc::gpuSetArg(g_sgemmKernel, 6, sizeof(float), alpha);
    xoc::gpuSetArg(g_sgemmKernel, 7, sizeof(float), beta);

    size_t gs[2] = { (size_t)((m + 15) / 16 * 16), (size_t)((n + 15) / 16 * 16) };
    size_t ls[2] = { 16, 16 };
    if (!xoc::gpuLaunch2D(g_sgemmKernel, gs, ls)) return xeCudaErrorInitializationError;
    xoc::gpuSync();

    return xeCudaSuccess;
}

xeCudaError_t xeCudaMatVecQ4KM(
    const uint8_t* q4_weights,
    const float* x_in,
    float* y_out,
    int rows,
    int cols
) {
    if (!q4_weights || !x_in || !y_out) return xeCudaErrorInvalidValue;
    if (!xoc::isInitialized()) return xeCudaErrorInitializationError;
    if (!ensureMatvecKernel()) return xeCudaErrorInitializationError;

    int halfCols = cols / 2;
    size_t sizeQ4 = (size_t)rows * halfCols;
    size_t sizeX = (size_t)cols * sizeof(float);
    size_t sizeY = (size_t)rows * sizeof(float);

    cl_mem dQ4 = xoc::gpuMalloc(sizeQ4);
    cl_mem dX  = xoc::gpuMalloc(sizeX);
    cl_mem dY  = xoc::gpuMalloc(sizeY);
    if (!dQ4 || !dX || !dY) {
        xoc::gpuFree(dQ4); xoc::gpuFree(dX); xoc::gpuFree(dY);
        return xeCudaErrorMemoryAllocation;
    }

    if (!xoc::gpuWrite(dQ4, q4_weights, sizeQ4) || !xoc::gpuWrite(dX, x_in, sizeX)) {
        xoc::gpuFree(dQ4); xoc::gpuFree(dX); xoc::gpuFree(dY);
        return xeCudaErrorInvalidMemcpyDirection;
    }

    cl_int rRows = rows, rCols = cols;
    xoc::gpuSetArg(g_matvecKernel, 0, sizeof(cl_mem), &dQ4);
    xoc::gpuSetArg(g_matvecKernel, 1, sizeof(cl_mem), &dX);
    xoc::gpuSetArg(g_matvecKernel, 2, sizeof(cl_mem), &dY);
    xoc::gpuSetArg(g_matvecKernel, 3, sizeof(cl_int), &rRows);
    xoc::gpuSetArg(g_matvecKernel, 4, sizeof(cl_int), &rCols);

    size_t gs = ((size_t)rows + 255) / 256 * 256;
    xoc::gpuLaunch(g_matvecKernel, gs, 256);
    xoc::gpuSync();

    if (!xoc::gpuRead(dY, y_out, sizeY)) {
        xoc::gpuFree(dQ4); xoc::gpuFree(dX); xoc::gpuFree(dY);
        return xeCudaErrorInvalidMemcpyDirection;
    }
    xoc::gpuFree(dQ4);
    xoc::gpuFree(dX);
    xoc::gpuFree(dY);
    return xeCudaSuccess;
}
