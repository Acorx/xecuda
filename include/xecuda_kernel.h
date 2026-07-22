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
    xeDim3 threadIdx;
    xeDim3 blockIdx;
    xeDim3 blockDim;
    xeDim3 gridDim;
};

// Global context thread local variable for kernel execution on Intel Arc
#ifdef __cplusplus
extern "C" {
#endif
extern thread_local xeWorkItemContext g_xeWorkContext;
#ifdef __cplusplus
}
#endif

#define threadIdx (g_xeWorkContext.threadIdx)
#define blockIdx  (g_xeWorkContext.blockIdx)
#define blockDim  (g_xeWorkContext.blockDim)
#define gridDim   (g_xeWorkContext.gridDim)

// Parallel Execution Launcher for Intel Arc Vector/Xe Cores
template <typename KernelFunc, typename... Args>
xeCudaError_t xeCudaLaunchKernel(KernelFunc kernel, xeDim3 grid, xeDim3 block, size_t sharedMem, xeCudaStream_t stream, Args... args) {
    (void)sharedMem;
    (void)stream;

#pragma omp parallel for collapse(3) schedule(dynamic) if(grid.x * grid.y * grid.z > 1)
    for (unsigned int gz = 0; gz < grid.z; ++gz) {
        for (unsigned int gy = 0; gy < grid.y; ++gy) {
            for (unsigned int gx = 0; gx < grid.x; ++gx) {
                for (unsigned int bz = 0; bz < block.z; ++bz) {
                    for (unsigned int by = 0; by < block.y; ++by) {
                        for (unsigned int bx = 0; bx < block.x; ++bx) {
                            g_xeWorkContext.gridDim = grid;
                            g_xeWorkContext.blockDim = block;
                            g_xeWorkContext.blockIdx = xeDim3(gx, gy, gz);
                            g_xeWorkContext.threadIdx = xeDim3(bx, by, bz);

                            kernel(args...);
                        }
                    }
                }
            }
        }
    }
    return xeCudaSuccess;
}

// Convenient Macro mimicking CUDA <<<grid, block>>> launcher
#define XECUDA_LAUNCH(kernel, grid, block, ...) \
    xeCudaLaunchKernel(kernel, grid, block, 0, nullptr, __VA_ARGS__)

#endif // XECUDA_KERNEL_H
