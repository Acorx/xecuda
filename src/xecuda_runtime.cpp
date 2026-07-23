/**
 * @file xecuda_runtime.cpp
 * @brief XeCUDA Runtime — Real GPU Memory & Synchronization via OpenCL
 */

#include "../include/xecuda_runtime.h"
#include "../include/xecuda_kernel.h"
#include "../include/xecuda_ocl.h"
#include <iostream>
#include <cstring>
#include <mutex>
#include <unordered_map>

// Map GPU buffer handles (cl_mem) to sizes for correct memcpy/memset
static std::unordered_map<void*, size_t> g_bufSizes;
static std::mutex g_bufMutex;

thread_local xeWorkItemContext g_xeWorkContext = {
    xeDim3(0, 0, 0), xeDim3(0, 0, 0), xeDim3(1, 1, 1), xeDim3(1, 1, 1)
};

static int g_currentDevice = 0;
static xeCudaError_t g_lastError = xeCudaSuccess;

struct xeCudaStream_st {
    int streamId;
};

// ─── Device Management ──────────────────────────────────────────

xeCudaError_t xeCudaGetDeviceCount(int* count) {
    if (!count) return xeCudaErrorInvalidValue;
    if (!xoc::isInitialized() && !xoc::init()) return xeCudaErrorInitializationError;
    *count = 1;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaSetDevice(int device) {
    if (device < 0 || device >= 1) return xeCudaErrorInvalidDevice;
    if (!xoc::isInitialized() && !xoc::init()) return xeCudaErrorInitializationError;
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
    if (!xoc::isInitialized() && !xoc::init()) return xeCudaErrorInitializationError;

    const auto& s = xoc::state();
    std::memset(prop, 0, sizeof(xeCudaDeviceProp_t));
    std::strncpy(prop->name, s.deviceName, sizeof(prop->name) - 1);
    prop->vendorId = 0x8086;
    prop->deviceId = 0x64A0;
    prop->totalGlobalMem = s.globalMemBytes;
    prop->xeCores = s.computeUnits;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->clockRateKHz = 1850000;
    prop->memoryClockRateKHz = 8533000;
    prop->memoryBusWidth = 128;
    prop->l2CacheSize = 8 * 1024 * 1024;
    prop->isIntegrated = 1;
    prop->supportsXMX = 1;
    return xeCudaSuccess;
}

// ─── GPU Memory Management (Real OpenCL cl_mem Buffers) ─────────

xeCudaError_t xeCudaMalloc(void** devPtr, size_t size) {
    if (!devPtr || size == 0) return xeCudaErrorInvalidValue;
    if (!xoc::isInitialized() && !xoc::init()) return xeCudaErrorInitializationError;

    cl_mem buf = xoc::gpuMalloc(size);
    if (!buf) return xeCudaErrorMemoryAllocation;

    {
        std::lock_guard<std::mutex> lock(g_bufMutex);
        g_bufSizes[buf] = size;
    }
    *devPtr = buf;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaMallocHost(void** ptr, size_t size) {
    // Host allocation is CPU-side — use aligned malloc for zero-copy
    if (!ptr || size == 0) return xeCudaErrorInvalidValue;
#if defined(_MSC_VER)
    *ptr = _aligned_malloc(size, 64);
#else
    *ptr = nullptr;
    if (posix_memalign(ptr, 64, size) != 0) *ptr = nullptr;
#endif
    return *ptr ? xeCudaSuccess : xeCudaErrorMemoryAllocation;
}

xeCudaError_t xeCudaFree(void* devPtr) {
    if (!devPtr) return xeCudaSuccess;
    if (!xoc::isInitialized()) {
        // During shutdown, just try to release
        xoc::gpuFree((cl_mem)devPtr);
        return xeCudaSuccess;
    }
    {
        std::lock_guard<std::mutex> lock(g_bufMutex);
        g_bufSizes.erase(devPtr);
    }
    xoc::gpuFree((cl_mem)devPtr);
    return xeCudaSuccess;
}

xeCudaError_t xeCudaFreeHost(void* ptr) {
    if (!ptr) return xeCudaSuccess;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
    return xeCudaSuccess;
}

xeCudaError_t xeCudaMemset(void* devPtr, int value, size_t count) {
    if (!devPtr) return xeCudaErrorInvalidValue;
    if (!xoc::isInitialized()) return xeCudaErrorInitializationError;

    // Build memset kernel on the fly
    static cl_program s_memsetProg = nullptr;
    static cl_kernel s_memsetKernel = nullptr;
    if (!s_memsetProg) {
        const char* src =
            "__kernel void memset_fill(__global uint* buf, const uint val, const int N) {\n"
            "    int i = get_global_id(0);\n"
            "    if (i < N) buf[i] = val;\n"
            "}\n";
        s_memsetProg = xoc::gpuCompileProgram(src, std::strlen(src));
        if (!s_memsetProg) return xeCudaErrorInitializationError;
        s_memsetKernel = xoc::gpuCreateKernel(s_memsetProg, "memset_fill");
        if (!s_memsetKernel) return xeCudaErrorInitializationError;
    }

    uint32_t fillVal = (uint32_t)(value & 0xFF) * 0x01010101u;
    uint32_t nWords = (uint32_t)(count / 4);

    cl_mem buf = (cl_mem)devPtr;
    xoc::gpuSetArg(s_memsetKernel, 0, sizeof(cl_mem), &buf);
    xoc::gpuSetArg(s_memsetKernel, 1, sizeof(uint32_t), &fillVal);
    xoc::gpuSetArg(s_memsetKernel, 2, sizeof(uint32_t), &nWords);

    size_t gs = ((nWords + 255) / 256) * 256;
    xoc::gpuLaunch(s_memsetKernel, gs, 256);
    xoc::gpuSync();
    return xeCudaSuccess;
}

xeCudaError_t xeCudaMemcpy(void* dst, const void* src, size_t count, xeCudaMemcpyKind kind) {
    if (!dst && !src) return xeCudaErrorInvalidValue;
    if (!xoc::isInitialized()) return xeCudaErrorInitializationError;

    switch (kind) {
    case xeCudaMemcpyHostToDevice:
        // src = CPU pointer, dst = cl_mem handle
        if (!xoc::gpuWrite((cl_mem)dst, src, count)) return xeCudaErrorInvalidMemcpyDirection;
        break;
    case xeCudaMemcpyDeviceToHost:
        // src = cl_mem handle, dst = CPU pointer
        if (!xoc::gpuRead((cl_mem)src, dst, count)) return xeCudaErrorInvalidMemcpyDirection;
        break;
    case xeCudaMemcpyDeviceToDevice: {
        // Allocate temp, copy GPU→CPU→GPU (true D2D needs clEnqueueCopyBuffer)
        void* tmp = std::malloc(count);
        if (!tmp) return xeCudaErrorMemoryAllocation;
        if (!xoc::gpuRead((cl_mem)src, tmp, count)) { std::free(tmp); return xeCudaErrorInvalidMemcpyDirection; }
        if (!xoc::gpuWrite((cl_mem)dst, tmp, count)) { std::free(tmp); return xeCudaErrorInvalidMemcpyDirection; }
        std::free(tmp);
        break;
    }
    case xeCudaMemcpyHostToHost:
        std::memcpy(dst, src, count);
        break;
    default:
        std::memcpy(dst, src, count);
        break;
    }
    return xeCudaSuccess;
}

xeCudaError_t xeCudaMemcpyAsync(void* dst, const void* src, size_t count, xeCudaMemcpyKind kind, xeCudaStream_t stream) {
    (void)stream;
    return xeCudaMemcpy(dst, src, count, kind);
}

// ─── Stream Management ──────────────────────────────────────────

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
    if (xoc::isInitialized()) xoc::gpuSync();
    return xeCudaSuccess;
}

xeCudaError_t xeCudaDeviceSynchronize(void) {
    if (xoc::isInitialized()) xoc::gpuSync();
    return xeCudaSuccess;
}

// ─── Error Handling ─────────────────────────────────────────────

const char* xeCudaGetErrorString(xeCudaError_t error) {
    switch (error) {
        case xeCudaSuccess: return "xeCudaSuccess: No error.";
        case xeCudaErrorInvalidValue: return "xeCudaErrorInvalidValue: Invalid parameter.";
        case xeCudaErrorMemoryAllocation: return "xeCudaErrorMemoryAllocation: GPU alloc failed.";
        case xeCudaErrorInitializationError: return "xeCudaErrorInitializationError: OpenCL init failed.";
        case xeCudaErrorInvalidDevice: return "xeCudaErrorInvalidDevice: Invalid device.";
        case xeCudaErrorInvalidMemcpyDirection: return "xeCudaErrorInvalidMemcpyDirection: Memcpy failed.";
        default: return "xeCudaErrorUnknown: Unknown error.";
    }
}

xeCudaError_t xeCudaGetLastError(void) {
    xeCudaError_t err = g_lastError;
    g_lastError = xeCudaSuccess;
    return err;
}
