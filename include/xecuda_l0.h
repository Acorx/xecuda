/**
 * @file xecuda_l0.h
 * @brief Bare-Metal Intel Level Zero (ze_api) Driver Interface for Intel Arc GPUs
 */

#ifndef XECUDA_L0_H
#define XECUDA_L0_H

#include "xecuda_runtime.h"
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Level Zero Handle Typdefs
typedef void* ze_driver_handle_t;
typedef void* ze_device_handle_t;
typedef void* ze_context_handle_t;
typedef void* ze_command_queue_handle_t;
typedef void* ze_command_list_handle_t;
typedef void* ze_module_handle_t;
typedef void* ze_kernel_handle_t;

// Level Zero Driver State Struct
typedef struct xeL0State {
    int initialized;
    ze_driver_handle_t driver;
    ze_device_handle_t device;
    ze_context_handle_t context;
    ze_command_queue_handle_t commandQueue;
    ze_command_list_handle_t commandList;
    uint32_t deviceId;
    char deviceName[256];
} xeL0State_t;

/**
 * @brief Initialize Bare-Metal Intel Level Zero Driver for Arc 130V
 */
xeCudaError_t xeL0Init(xeL0State_t* state);

/**
 * @brief Allocate Zero-Copy Unified Shared Memory (USM) on Lunar Lake
 */
xeCudaError_t xeL0AllocSharedMemory(xeL0State_t* state, size_t size, void** ptr);

/**
 * @brief Free Level Zero USM Memory
 */
xeCudaError_t xeL0FreeMemory(xeL0State_t* state, void* ptr);

/**
 * @brief Create Level Zero SPIR-V Execution Module
 */
xeCudaError_t xeL0CreateModuleFromSpirv(
    xeL0State_t* state,
    const uint8_t* spirvCode,
    size_t spirvSize,
    const char* kernelName,
    ze_module_handle_t* module,
    ze_kernel_handle_t* kernel
);

/**
 * @brief Bare-Metal Kernel Launch directly on Xe2 Vector Cores
 */
xeCudaError_t xeL0LaunchKernel(
    xeL0State_t* state,
    ze_kernel_handle_t kernel,
    uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
    uint32_t groupSizeX, uint32_t groupSizeY, uint32_t groupSizeZ
);

/**
 * @brief Cleanup Level Zero Driver Resources
 */
xeCudaError_t xeL0Shutdown(xeL0State_t* state);

#ifdef __cplusplus
}
#endif

#endif // XECUDA_L0_H
