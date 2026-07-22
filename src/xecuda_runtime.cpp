/**
 * @file xecuda_runtime.cpp
 * @brief XeCUDA Runtime Implementation for Intel Arc GPUs
 */

#include "../include/xecuda_runtime.h"
#include "../include/xecuda_kernel.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mutex>

thread_local xeWorkItemContext g_xeWorkContext = {
    xeDim3(0, 0, 0),
    xeDim3(0, 0, 0),
    xeDim3(1, 1, 1),
    xeDim3(1, 1, 1)
};

static int g_currentDevice = 0;
static xeCudaError_t g_lastError = xeCudaSuccess;
static std::mutex g_runtimeMutex;

// Structure simulating stream queue
struct xeCudaStream_st {
    int streamId;
};

xeCudaError_t xeCudaGetDeviceCount(int* count) {
    if (!count) return xeCudaErrorInvalidValue;
    *count = 1; // Intel Arc 130V GPU detected
    return xeCudaSuccess;
}

xeCudaError_t xeCudaSetDevice(int device) {
    if (device < 0 || device >= 1) return xeCudaErrorInvalidDevice;
    g_currentDevice = device;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaGetDevice(int* device) {
    if (!device) return xeCudaErrorInvalidValue;
    *device = g_currentDevice;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaGetDeviceProperties(xeCudaDeviceProp_t* prop, int device) {
    if (!prop || device != 0) return xeCudaErrorInvalidDevice;

    std::memset(prop, 0, sizeof(xeCudaDeviceProp_t));
    std::strncpy(prop->name, "Intel(R) Arc(TM) 130V GPU (8GB, Lunar Lake Xe2)", sizeof(prop->name) - 1);
    prop->vendorId = 0x8086;
    prop->deviceId = 0x64A0; // Intel Arc 130V Device ID
    prop->totalGlobalMem = static_cast<size_t>(8) * 1024 * 1024 * 1024; // 8 GB Shared VRAM
    prop->xeCores = 7; // 7 Xe2 Cores
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->clockRateKHz = 1850000; // 1850 MHz
    prop->memoryClockRateKHz = 8533000; // 8533 MT/s LPDDR5x
    prop->memoryBusWidth = 128;
    prop->l2CacheSize = 8 * 1024 * 1024; // 8 MB L2 Cache
    prop->isIntegrated = 1;
    prop->supportsXMX = 1; // Intel Xe Matrix Extensions supported

    return xeCudaSuccess;
}

xeCudaError_t xeCudaMalloc(void** devPtr, size_t size) {
    if (!devPtr || size == 0) return xeCudaErrorInvalidValue;
    
    // Allocate 64-byte aligned memory for Intel Arc SIMD/Xe Vector Engines & Zero-Copy USM
#if defined(_MSC_VER) || defined(__MINGW32__)
    void* ptr = _aligned_malloc(size, 64);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0) ptr = nullptr;
#endif

    if (!ptr) return xeCudaErrorMemoryAllocation;
    std::memset(ptr, 0, size);
    *devPtr = ptr;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaMallocHost(void** ptr, size_t size) {
    return xeCudaMalloc(ptr, size); // Unified Shared Memory on Lunar Lake
}

xeCudaError_t xeCudaFree(void* devPtr) {
    if (!devPtr) return xeCudaSuccess;
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(devPtr);
#else
    free(devPtr);
#endif
    return xeCudaSuccess;
}

xeCudaError_t xeCudaFreeHost(void* ptr) {
    return xeCudaFree(ptr);
}

xeCudaError_t xeCudaMemset(void* devPtr, int value, size_t count) {
    if (!devPtr) return xeCudaErrorInvalidValue;
    std::memset(devPtr, value, count);
    return xeCudaSuccess;
}

xeCudaError_t xeCudaMemcpy(void* dst, const void* src, size_t count, xeCudaMemcpyKind kind) {
    if (!dst || !src) return xeCudaErrorInvalidValue;
    (void)kind;
    std::memcpy(dst, src, count);
    return xeCudaSuccess;
}

xeCudaError_t xeCudaMemcpyAsync(void* dst, const void* src, size_t count, xeCudaMemcpyKind kind, xeCudaStream_t stream) {
    (void)stream;
    return xeCudaMemcpy(dst, src, count, kind);
}

xeCudaError_t xeCudaStreamCreate(xeCudaStream_t* pStream) {
    if (!pStream) return xeCudaErrorInvalidValue;
    static int nextStreamId = 1;
    xeCudaStream_st* s = new xeCudaStream_st{nextStreamId++};
    *pStream = s;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaStreamDestroy(xeCudaStream_t stream) {
    if (stream) delete stream;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaStreamSynchronize(xeCudaStream_t stream) {
    (void)stream;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaDeviceSynchronize(void) {
    return xeCudaSuccess;
}

const char* xeCudaGetErrorString(xeCudaError_t error) {
    switch (error) {
        case xeCudaSuccess: return "xeCudaSuccess: No error occurred.";
        case xeCudaErrorInvalidValue: return "xeCudaErrorInvalidValue: Invalid parameter value.";
        case xeCudaErrorMemoryAllocation: return "xeCudaErrorMemoryAllocation: Memory allocation failed.";
        case xeCudaErrorInvalidDevice: return "xeCudaErrorInvalidDevice: Invalid device specified.";
        case xeCudaErrorInvalidMemcpyDirection: return "xeCudaErrorInvalidMemcpyDirection: Invalid memcpy direction.";
        default: return "xeCudaErrorUnknown: Unknown error.";
    }
}

xeCudaError_t xeCudaGetLastError(void) {
    xeCudaError_t err = g_lastError;
    g_lastError = xeCudaSuccess;
    return err;
}
