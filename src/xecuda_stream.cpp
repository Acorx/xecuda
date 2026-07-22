/**
 * @file xecuda_stream.cpp
 * @brief CUDA Stream & Event Implementation for Intel Arc
 */

#include "../include/xecuda_stream.h"
#include <chrono>
#include <iostream>

struct xeCudaEvent_st {
    std::chrono::high_resolution_clock::time_point timeStamp;
};

xeCudaError_t xeCudaEventCreate(xeCudaEvent_t* pEvent) {
    if (!pEvent) return xeCudaErrorInvalidValue;
    xeCudaEvent_st* ev = new xeCudaEvent_st();
    *pEvent = ev;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaEventDestroy(xeCudaEvent_t event) {
    if (event) delete event;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaEventRecord(xeCudaEvent_t event, xeCudaStream_t stream) {
    (void)stream;
    if (!event) return xeCudaErrorInvalidValue;
    event->timeStamp = std::chrono::high_resolution_clock::now();
    return xeCudaSuccess;
}

xeCudaError_t xeCudaEventSynchronize(xeCudaEvent_t event) {
    (void)event;
    return xeCudaSuccess;
}

xeCudaError_t xeCudaEventElapsedTime(float* ms, xeCudaEvent_t start, xeCudaEvent_t end) {
    if (!ms || !start || !end) return xeCudaErrorInvalidValue;
    double duration = std::chrono::duration<double, std::milli>(end->timeStamp - start->timeStamp).count();
    *ms = static_cast<float>(duration);
    return xeCudaSuccess;
}
