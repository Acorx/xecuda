/**
 * @file xecuda_runtime.h
 * @brief CUDA Runtime API Translation Layer for Intel Arc GPUs (Level Zero / SYCL Backend)
 */

#ifndef XECUDA_RUNTIME_H
#define XECUDA_RUNTIME_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Error Codes
typedef enum xeCudaError {
    xeCudaSuccess = 0,
    xeCudaErrorInvalidValue = 1,
    xeCudaErrorMemoryAllocation = 2,
    xeCudaErrorInitializationError = 3,
    xeCudaErrorDeinitialized = 4,
    xeCudaErrorInvalidDevice = 5,
    xeCudaErrorInvalidMemcpyDirection = 6,
    xeCudaErrorNotYetImplemented = 999
} xeCudaError_t;

// Memcpy Directions
typedef enum xeCudaMemcpyKind {
    xeCudaMemcpyHostToHost = 0,
    xeCudaMemcpyHostToDevice = 1,
    xeCudaMemcpyDeviceToHost = 2,
    xeCudaMemcpyDeviceToDevice = 3,
    xeCudaMemcpyDefault = 4
} xeCudaMemcpyKind;

// Device Properties Struct
typedef struct xeCudaDeviceProp {
    char name[256];
    uint32_t vendorId;
    uint32_t deviceId;
    size_t totalGlobalMem;
    int xeCores;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRateKHz;
    int memoryClockRateKHz;
    int memoryBusWidth;
    int l2CacheSize;
    int isIntegrated;
    int supportsXMX; // Xe Matrix Extensions (XMX / DP4a)
} xeCudaDeviceProp_t;

// Stream Handle
typedef struct xeCudaStream_st* xeCudaStream_t;

// Device Management
xeCudaError_t xeCudaGetDeviceCount(int* count);
xeCudaError_t xeCudaSetDevice(int device);
xeCudaError_t xeCudaGetDevice(int* device);
xeCudaError_t xeCudaGetDeviceProperties(xeCudaDeviceProp_t* prop, int device);

// Memory Management
xeCudaError_t xeCudaMalloc(void** devPtr, size_t size);
xeCudaError_t xeCudaMallocHost(void** ptr, size_t size);
xeCudaError_t xeCudaFree(void* devPtr);
xeCudaError_t xeCudaFreeHost(void* ptr);
xeCudaError_t xeCudaMemset(void* devPtr, int value, size_t count);
xeCudaError_t xeCudaMemcpy(void* dst, const void* src, size_t count, xeCudaMemcpyKind kind);
xeCudaError_t xeCudaMemcpyAsync(void* dst, const void* src, size_t count, xeCudaMemcpyKind kind, xeCudaStream_t stream);

// Stream Management
xeCudaError_t xeCudaStreamCreate(xeCudaStream_t* pStream);
xeCudaError_t xeCudaStreamDestroy(xeCudaStream_t stream);
xeCudaError_t xeCudaStreamSynchronize(xeCudaStream_t stream);
xeCudaError_t xeCudaDeviceSynchronize(void);

// Error Handling
const char* xeCudaGetErrorString(xeCudaError_t error);
xeCudaError_t xeCudaGetLastError(void);

#ifdef __cplusplus
}
#endif

#endif // XECUDA_RUNTIME_H
