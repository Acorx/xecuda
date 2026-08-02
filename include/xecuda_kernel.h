/**
 * @file xecuda_kernel.h
 * @brief Kernel Execution & Thread Index Emulation Header for Intel Arc
 */

#ifndef XECUDA_KERNEL_H
#define XECUDA_KERNEL_H

#include "xecuda_runtime.h"

// Dimension Struct (equivalent to CUDA dim3)
struct xeDim3 {
    unsigned int x, y, z;
    xeDim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1) : x(_x), y(_y), z(_z) {}
};
typedef struct xeDim3 xeDim3;

// CUDA Qualifier Shims
// NOTE: __shared__ maps to thread_local (each thread gets a private copy).
// True shared memory (block-shared) is not yet supported on the CPU emulation path.
#ifndef __global__
#define __global__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __shared__
#define __shared__ thread_local
#endif

// Thread Index Context Struct passed during kernel execution
struct xeWorkItemContext {
    xeDim3 tid;
    xeDim3 bid;
    xeDim3 bdim;
    xeDim3 gdim;
};

// Global context thread local variable for kernel execution on Intel Arc
#ifdef __cplusplus
extern "C" {
#endif
extern thread_local xeWorkItemContext g_xeWorkContext;
#ifdef __cplusplus
}
#endif

#define threadIdx (g_xeWorkContext.tid)
#define blockIdx  (g_xeWorkContext.bid)
#define blockDim  (g_xeWorkContext.bdim)
#define gridDim   (g_xeWorkContext.gdim)

// Parallel Execution Launcher for Intel Arc Vector/Xe Cores
template <typename KernelFunc, typename... Args>
xeCudaError_t xeCudaLaunchKernel(KernelFunc kernel, xeDim3 grid, xeDim3 block, size_t sharedMem, xeCudaStream_t stream, Args... args) {
    (void)sharedMem;
    (void)stream;

    unsigned long long totalThreads = (unsigned long long)grid.x * grid.y * grid.z * block.x * block.y * block.z;

    if (totalThreads <= 1) {
        // Single-thread fallback: no OpenMP overhead
        g_xeWorkContext.gdim = grid;
        g_xeWorkContext.bdim = block;
        g_xeWorkContext.bid = xeDim3(0, 0, 0);
        g_xeWorkContext.tid = xeDim3(0, 0, 0);
        kernel(args...);
    } else {
        // Each OpenMP thread processes one (grid, block) work-item
        // Flatten all thread indices into a single loop for correct collapse
        const unsigned int gx_max = grid.x;
        const unsigned int gy_max = grid.y;
        const unsigned int gz_max = grid.z;
        const unsigned int bx_max = block.x;
        const unsigned int by_max = block.y;
        const unsigned int bz_max = block.z;
        const unsigned long long flat_max = (unsigned long long)gx_max * gy_max * gz_max * bx_max * by_max * bz_max;

#pragma omp parallel for schedule(dynamic)
        for (unsigned long long flat = 0; flat < flat_max; ++flat) {
            unsigned int bx = flat % bx_max;
            unsigned int by = (flat / bx_max) % by_max;
            unsigned int bz = (flat / (bx_max * by_max)) % bz_max;
            unsigned int gx = (flat / (bx_max * by_max * bz_max)) % gx_max;
            unsigned int gy = (flat / (bx_max * by_max * bz_max * gx_max)) % gy_max;
            unsigned int gz = (flat / (bx_max * by_max * bz_max * gx_max * gy_max)) % gz_max;

            g_xeWorkContext.gdim = grid;
            g_xeWorkContext.bdim = block;
            g_xeWorkContext.bid = xeDim3(gx, gy, gz);
            g_xeWorkContext.tid = xeDim3(bx, by, bz);

            kernel(args...);
        }
    }
    return xeCudaSuccess;
}

// Convenient Macro mimicking CUDA <<<grid, block>>> launcher
#define XECUDA_LAUNCH(kernel, grid, block, ...) \
    xeCudaLaunchKernel(kernel, grid, block, 0, nullptr, __VA_ARGS__)

#endif // XECUDA_KERNEL_H
