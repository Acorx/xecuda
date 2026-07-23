/**
 * @file xecuda_l0.cpp
 * @brief Level Zero Driver Backend — Real GPU dispatch via OpenCL
 */

#include "../include/xecuda_l0.h"
#include "../include/xecuda_ocl.h"
#include <iostream>
#include <cstring>
#include <mutex>
#include <vector>

// ============================================================================
// OpenCL-based Level Zero backend
// ============================================================================
// Level Zero and OpenCL are both supported by the Intel NEO driver.
// This implementation maps L0 semantics onto real OpenCL API calls
// so that the xecuda_l0.h API surface works with actual GPU dispatch.

static std::vector<cl_program> g_l0Programs;
static std::vector<cl_kernel> g_l0Kernels;
static std::mutex g_l0Mutex;

xeCudaError_t xeL0Init(xeL0State_t* state) {
    if (!state) return xeCudaErrorInvalidValue;

    std::memset(state, 0, sizeof(xeL0State_t));

    if (!xoc::isInitialized() && !xoc::init()) {
        std::cerr << "[XeCUDA L0] Failed to initialize OpenCL backend." << std::endl;
        return xeCudaErrorInitializationError;
    }

    const auto& s = xoc::state();

    state->initialized = 1;
    state->driver = (ze_driver_handle_t)1;
    state->device = (ze_device_handle_t)2;
    state->context = (ze_context_handle_t)3;
    state->commandQueue = (ze_command_queue_handle_t)4;
    state->commandList = (ze_command_list_handle_t)5;
    state->deviceId = 0x64A0;
    std::strncpy(state->deviceName, s.deviceName, sizeof(state->deviceName) - 1);

    std::cout << "[XeCUDA L0] Real GPU backend via OpenCL 3.0 Intel NEO." << std::endl;
    std::cout << "            Device: " << s.deviceName << " (" << s.computeUnits << " CU)" << std::endl;
    std::cout << "            VRAM: " << (s.globalMemBytes / (1024ULL * 1024ULL)) << " MB" << std::endl;

    return xeCudaSuccess;
}

xeCudaError_t xeL0AllocSharedMemory(xeL0State_t* state, size_t size, void** ptr) {
    if (!state || !state->initialized || !ptr || size == 0) return xeCudaErrorInvalidValue;
    if (!xoc::isInitialized()) return xeCudaErrorInitializationError;

    cl_mem buf = xoc::gpuMalloc(size);
    if (!buf) return xeCudaErrorMemoryAllocation;

    *ptr = buf;
    return xeCudaSuccess;
}

xeCudaError_t xeL0FreeMemory(xeL0State_t* state, void* ptr) {
    if (!state || !state->initialized) return xeCudaErrorInvalidValue;
    xoc::gpuFree((cl_mem)ptr);
    return xeCudaSuccess;
}

xeCudaError_t xeL0CreateModuleFromSpirv(
    xeL0State_t* state,
    const uint8_t* spirvCode,
    size_t spirvSize,
    const char* kernelName,
    ze_module_handle_t* module,
    ze_kernel_handle_t* kernel
) {
    if (!state || !state->initialized || !spirvCode || spirvSize == 0 || !kernelName || !module || !kernel) {
        return xeCudaErrorInvalidValue;
    }
    if (!xoc::isInitialized()) return xeCudaErrorInitializationError;

    // The spirvCode is actually OpenCL C source text (from xecudac translator)
    cl_program prog = xoc::gpuCompileProgram((const char*)spirvCode, spirvSize);
    if (!prog) {
        std::cerr << "[XeCUDA L0] Kernel build failed for '" << kernelName << "'" << std::endl;
        return xeCudaErrorInitializationError;
    }

    cl_kernel kern = xoc::gpuCreateKernel(prog, kernelName);
    if (!kern) {
        xoc::gpuReleaseProgram(prog);
        std::cerr << "[XeCUDA L0] clCreateKernel failed for '" << kernelName << "'" << std::endl;
        return xeCudaErrorInitializationError;
    }

    {
        std::lock_guard<std::mutex> lock(g_l0Mutex);
        g_l0Programs.push_back(prog);
        g_l0Kernels.push_back(kern);
    }

    *module = (ze_module_handle_t)prog;
    *kernel = (ze_kernel_handle_t)kern;

    std::cout << "[XeCUDA L0] Compiled and loaded kernel '" << kernelName
              << "' (" << spirvSize << " bytes source)" << std::endl;
    return xeCudaSuccess;
}

xeCudaError_t xeL0LaunchKernel(
    xeL0State_t* state,
    ze_kernel_handle_t kernel,
    uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
    uint32_t groupSizeX, uint32_t groupSizeY, uint32_t groupSizeZ
) {
    if (!state || !state->initialized || !kernel) return xeCudaErrorInvalidValue;
    if (!xoc::isInitialized()) return xeCudaErrorInitializationError;

    // 1D launch: flatten all groups
    size_t globalSize = (size_t)groupCountX * groupSizeX;
    size_t localSize = (size_t)groupSizeX;

    cl_kernel kern = (cl_kernel)kernel;
    if (!xoc::gpuLaunch(kern, globalSize, localSize)) {
        return xeCudaErrorInitializationError;
    }
    xoc::gpuSync();

    return xeCudaSuccess;
}

xeCudaError_t xeL0Shutdown(xeL0State_t* state) {
    if (!state) return xeCudaErrorInvalidValue;

    {
        std::lock_guard<std::mutex> lock(g_l0Mutex);
        for (cl_kernel k : g_l0Kernels) xoc::gpuReleaseKernel(k);
        for (cl_program p : g_l0Programs) xoc::gpuReleaseProgram(p);
        g_l0Kernels.clear();
        g_l0Programs.clear();
    }

    xoc::shutdown();
    state->initialized = 0;
    std::cout << "[XeCUDA L0] GPU backend shutdown complete." << std::endl;
    return xeCudaSuccess;
}
