/**
 * @file xecuda_ze_loader.h
 * @brief Dynamic Win32 Loader for Intel Level Zero (ze_api.dll / ze_loader.dll)
 */

#ifndef XECUDA_ZE_LOADER_H
#define XECUDA_ZE_LOADER_H

#include "xecuda_runtime.h"
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xeZeDriverHandle {
    void* hDll;
    int isLoaded;
    void* driverHandle;
    void* deviceHandle;
    void* contextHandle;
    uint32_t deviceId;
} xeZeDriverHandle_t;

/**
 * @brief Dynamically load C:\Windows\System32\ze_loader.dll and initialize Level Zero
 */
xeCudaError_t xeZeLoadSystemDriver(xeZeDriverHandle_t* handle);

/**
 * @brief Allocate Zero-Copy VRAM memory via real zeMemAllocShared API
 */
xeCudaError_t xeZeAllocSharedVRAM(xeZeDriverHandle_t* handle, size_t size, void** ptr);

/**
 * @brief Unload Level Zero driver
 */
xeCudaError_t xeZeUnloadSystemDriver(xeZeDriverHandle_t* handle);

#ifdef __cplusplus
}
#endif

#endif // XECUDA_ZE_LOADER_H
