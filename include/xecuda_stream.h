/**
 * @file xecuda_stream.h
 * @brief CUDA Stream & Event Synchronization Engine for Intel Arc GPUs
 */

#ifndef XECUDA_STREAM_H
#define XECUDA_STREAM_H

#include "xecuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xeCudaEvent_st* xeCudaEvent_t;

// Event Management
xeCudaError_t xeCudaEventCreate(xeCudaEvent_t* pEvent);
xeCudaError_t xeCudaEventDestroy(xeCudaEvent_t event);
xeCudaError_t xeCudaEventRecord(xeCudaEvent_t event, xeCudaStream_t stream);
xeCudaError_t xeCudaEventSynchronize(xeCudaEvent_t event);
xeCudaError_t xeCudaEventElapsedTime(float* ms, xeCudaEvent_t start, xeCudaEvent_t end);

#ifdef __cplusplus
}
#endif

#endif // XECUDA_STREAM_H
