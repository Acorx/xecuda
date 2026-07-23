/**
 * @file xecuda_ocl.h
 * @brief OpenCL Backend Singleton for Intel Arc 130V GPU
 *        Loaded at runtime from OpenCL.dll via LoadLibrary/GetProcAddress.
 *        ALL GPU memory allocation, kernel compilation, and dispatch flows through here.
 */

#ifndef XECUDA_OCL_H
#define XECUDA_OCL_H

#include <cstdint>
#include <cstddef>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// ============================================================================
// Minimal OpenCL type definitions (avoids dependency on CL/cl.h)
// ============================================================================

typedef int            cl_int;
typedef unsigned int   cl_uint;
typedef size_t         cl_size_t;
typedef void*          cl_platform_id;
typedef void*          cl_device_id;
typedef void*          cl_context;
typedef void*          cl_command_queue;
typedef void*          cl_mem;
typedef void*          cl_program;
typedef void*          cl_kernel;
typedef void*          cl_event;

#define CL_SUCCESS                     0
#define CL_DEVICE_TYPE_ALL             0xFFFFFFFF
#define CL_MEM_READ_WRITE              0x0001
#define CL_MEM_COPY_HOST_PTR           0x0004

// Device info queries
#define CL_DEVICE_NAME                 0x102F
#define CL_DEVICE_VENDOR               0x102C
#define CL_DRIVER_VERSION              0x1097
#define CL_DEVICE_MAX_COMPUTE_UNITS    0x1002
#define CL_DEVICE_GLOBAL_MEM_SIZE      0x101F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE   0x1010

// Program build info
#define CL_PROGRAM_BUILD_LOG           0x1184

// ============================================================================
// OpenCL function pointer typedefs
// ============================================================================

typedef cl_int (CL_API_CALL *PFN_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CL_API_CALL *PFN_clGetDeviceIDs)(cl_platform_id, cl_uint, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (CL_API_CALL *PFN_clGetDeviceInfo)(cl_device_id, cl_uint, cl_size_t, void*, cl_size_t*);
typedef cl_context (CL_API_CALL *PFN_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (CL_API_CALL *PFN_clCreateCommandQueue)(cl_context, cl_device_id, cl_uint, cl_int*);
typedef cl_mem (CL_API_CALL *PFN_clCreateBuffer)(cl_context, cl_uint, cl_size_t, void*, cl_int*);
typedef cl_program (CL_API_CALL *PFN_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const cl_size_t*, cl_int*);
typedef cl_int (CL_API_CALL *PFN_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_int (CL_API_CALL *PFN_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, cl_size_t, void*, cl_size_t*);
typedef cl_kernel (CL_API_CALL *PFN_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (CL_API_CALL *PFN_clSetKernelArg)(cl_kernel, cl_uint, cl_size_t, const void*);
typedef cl_int (CL_API_CALL *PFN_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const cl_size_t*, const cl_size_t*, const cl_size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL *PFN_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, cl_size_t, cl_size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL *PFN_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, cl_size_t, cl_size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL *PFN_clFinish)(cl_command_queue);
typedef cl_int (CL_API_CALL *PFN_clReleaseMemObject)(cl_mem);
typedef cl_int (CL_API_CALL *PFN_clReleaseKernel)(cl_kernel);
typedef cl_int (CL_API_CALL *PFN_clReleaseProgram)(cl_program);
typedef cl_int (CL_API_CALL *PFN_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (CL_API_CALL *PFN_clReleaseContext)(cl_context);

// ============================================================================
// XeCUDA OpenCL Backend Singleton
// ============================================================================

namespace xoc {

struct BackendState {
    bool initialized;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    char deviceName[256];
    char driverVersion[256];
    uint32_t computeUnits;
    uint64_t globalMemBytes;
    uint64_t maxAllocBytes;
};

// All OpenCL function pointers
struct OCLFuncs {
    PFN_clGetPlatformIDs        GetPlatformIDs;
    PFN_clGetDeviceIDs          GetDeviceIDs;
    PFN_clGetDeviceInfo         GetDeviceInfo;
    PFN_clCreateContext         CreateContext;
    PFN_clCreateCommandQueue    CreateCommandQueue;
    PFN_clCreateBuffer          CreateBuffer;
    PFN_clCreateProgramWithSource CreateProgramWithSource;
    PFN_clBuildProgram          BuildProgram;
    PFN_clGetProgramBuildInfo   GetProgramBuildInfo;
    PFN_clCreateKernel          CreateKernel;
    PFN_clSetKernelArg          SetKernelArg;
    PFN_clEnqueueNDRangeKernel  EnqueueNDRangeKernel;
    PFN_clEnqueueWriteBuffer    EnqueueWriteBuffer;
    PFN_clEnqueueReadBuffer     EnqueueReadBuffer;
    PFN_clFinish                Finish;
    PFN_clReleaseMemObject      ReleaseMemObject;
    PFN_clReleaseKernel         ReleaseKernel;
    PFN_clReleaseProgram        ReleaseProgram;
    PFN_clReleaseCommandQueue   ReleaseCommandQueue;
    PFN_clReleaseContext        ReleaseContext;
};

static BackendState g_state = {};
static OCLFuncs g_func = {};
static HMODULE g_hCL = nullptr;

inline bool init() {
    if (g_state.initialized) return true;

#ifdef _WIN32
    g_hCL = LoadLibraryA("OpenCL.dll");
    if (!g_hCL) return false;

    g_func.GetPlatformIDs        = (PFN_clGetPlatformIDs)GetProcAddress(g_hCL, "clGetPlatformIDs");
    g_func.GetDeviceIDs          = (PFN_clGetDeviceIDs)GetProcAddress(g_hCL, "clGetDeviceIDs");
    g_func.GetDeviceInfo         = (PFN_clGetDeviceInfo)GetProcAddress(g_hCL, "clGetDeviceInfo");
    g_func.CreateContext         = (PFN_clCreateContext)GetProcAddress(g_hCL, "clCreateContext");
    g_func.CreateCommandQueue    = (PFN_clCreateCommandQueue)GetProcAddress(g_hCL, "clCreateCommandQueue");
    g_func.CreateBuffer          = (PFN_clCreateBuffer)GetProcAddress(g_hCL, "clCreateBuffer");
    g_func.CreateProgramWithSource = (PFN_clCreateProgramWithSource)GetProcAddress(g_hCL, "clCreateProgramWithSource");
    g_func.BuildProgram          = (PFN_clBuildProgram)GetProcAddress(g_hCL, "clBuildProgram");
    g_func.GetProgramBuildInfo   = (PFN_clGetProgramBuildInfo)GetProcAddress(g_hCL, "clGetProgramBuildInfo");
    g_func.CreateKernel          = (PFN_clCreateKernel)GetProcAddress(g_hCL, "clCreateKernel");
    g_func.SetKernelArg          = (PFN_clSetKernelArg)GetProcAddress(g_hCL, "clSetKernelArg");
    g_func.EnqueueNDRangeKernel  = (PFN_clEnqueueNDRangeKernel)GetProcAddress(g_hCL, "clEnqueueNDRangeKernel");
    g_func.EnqueueWriteBuffer    = (PFN_clEnqueueWriteBuffer)GetProcAddress(g_hCL, "clEnqueueWriteBuffer");
    g_func.EnqueueReadBuffer     = (PFN_clEnqueueReadBuffer)GetProcAddress(g_hCL, "clEnqueueReadBuffer");
    g_func.Finish                = (PFN_clFinish)GetProcAddress(g_hCL, "clFinish");
    g_func.ReleaseMemObject      = (PFN_clReleaseMemObject)GetProcAddress(g_hCL, "clReleaseMemObject");
    g_func.ReleaseKernel         = (PFN_clReleaseKernel)GetProcAddress(g_hCL, "clReleaseKernel");
    g_func.ReleaseProgram        = (PFN_clReleaseProgram)GetProcAddress(g_hCL, "clReleaseProgram");
    g_func.ReleaseCommandQueue   = (PFN_clReleaseCommandQueue)GetProcAddress(g_hCL, "clReleaseCommandQueue");
    g_func.ReleaseContext        = (PFN_clReleaseContext)GetProcAddress(g_hCL, "clReleaseContext");
#else
    g_hCL = dlopen("libOpenCL.so.1", RTLD_LAZY);
    if (!g_hCL) g_hCL = dlopen("libOpenCL.so", RTLD_LAZY);
    if (!g_hCL) return false;

    g_func.GetPlatformIDs        = (PFN_clGetPlatformIDs)dlsym(g_hCL, "clGetPlatformIDs");
    g_func.GetDeviceIDs          = (PFN_clGetDeviceIDs)dlsym(g_hCL, "clGetDeviceIDs");
    g_func.GetDeviceInfo         = (PFN_clGetDeviceInfo)dlsym(g_hCL, "clGetDeviceInfo");
    g_func.CreateContext         = (PFN_clCreateContext)dlsym(g_hCL, "clCreateContext");
    g_func.CreateCommandQueue    = (PFN_clCreateCommandQueue)dlsym(g_hCL, "clCreateCommandQueue");
    g_func.CreateBuffer          = (PFN_clCreateBuffer)dlsym(g_hCL, "clCreateBuffer");
    g_func.CreateProgramWithSource = (PFN_clCreateProgramWithSource)dlsym(g_hCL, "clCreateProgramWithSource");
    g_func.BuildProgram          = (PFN_clBuildProgram)dlsym(g_hCL, "clBuildProgram");
    g_func.GetProgramBuildInfo   = (PFN_clGetProgramBuildInfo)dlsym(g_hCL, "clGetProgramBuildInfo");
    g_func.CreateKernel          = (PFN_clCreateKernel)dlsym(g_hCL, "clCreateKernel");
    g_func.SetKernelArg          = (PFN_clSetKernelArg)dlsym(g_hCL, "clSetKernelArg");
    g_func.EnqueueNDRangeKernel  = (PFN_clEnqueueNDRangeKernel)dlsym(g_hCL, "clEnqueueNDRangeKernel");
    g_func.EnqueueWriteBuffer    = (PFN_clEnqueueWriteBuffer)dlsym(g_hCL, "clEnqueueWriteBuffer");
    g_func.EnqueueReadBuffer     = (PFN_clEnqueueReadBuffer)dlsym(g_hCL, "clEnqueueReadBuffer");
    g_func.Finish                = (PFN_clFinish)dlsym(g_hCL, "clFinish");
    g_func.ReleaseMemObject      = (PFN_clReleaseMemObject)dlsym(g_hCL, "clReleaseMemObject");
    g_func.ReleaseKernel         = (PFN_clReleaseKernel)dlsym(g_hCL, "clReleaseKernel");
    g_func.ReleaseProgram        = (PFN_clReleaseProgram)dlsym(g_hCL, "clReleaseProgram");
    g_func.ReleaseCommandQueue   = (PFN_clReleaseCommandQueue)dlsym(g_hCL, "clReleaseCommandQueue");
    g_func.ReleaseContext        = (PFN_clReleaseContext)dlsym(g_hCL, "clReleaseContext");
#endif

    if (!g_func.GetPlatformIDs || !g_func.CreateContext) return false;

    cl_int err;
    cl_uint nPlat = 0;
    g_func.GetPlatformIDs(0, nullptr, &nPlat);
    if (nPlat == 0) return false;
    cl_platform_id* platforms = new cl_platform_id[nPlat];
    g_func.GetPlatformIDs(nPlat, platforms, nullptr);
    g_state.platform = platforms[0];
    delete[] platforms;

    cl_uint nDev = 0;
    g_func.GetDeviceIDs(g_state.platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &nDev);
    if (nDev == 0) return false;
    cl_device_id* devs = new cl_device_id[nDev];
    g_func.GetDeviceIDs(g_state.platform, CL_DEVICE_TYPE_ALL, nDev, devs, nullptr);
    g_state.device = devs[0];
    delete[] devs;

    g_func.GetDeviceInfo(g_state.device, CL_DEVICE_NAME, sizeof(g_state.deviceName), g_state.deviceName, nullptr);
    g_func.GetDeviceInfo(g_state.device, CL_DRIVER_VERSION, sizeof(g_state.driverVersion), g_state.driverVersion, nullptr);
    g_func.GetDeviceInfo(g_state.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(g_state.computeUnits), &g_state.computeUnits, nullptr);
    g_func.GetDeviceInfo(g_state.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(g_state.globalMemBytes), &g_state.globalMemBytes, nullptr);
    g_func.GetDeviceInfo(g_state.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(g_state.maxAllocBytes), &g_state.maxAllocBytes, nullptr);

    cl_device_id devArr[1] = { g_state.device };
    g_state.context = g_func.CreateContext(nullptr, 1, devArr, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    g_state.queue = g_func.CreateCommandQueue(g_state.context, g_state.device, 0, &err);
    if (err != CL_SUCCESS) return false;

    g_state.initialized = true;
    return true;
}

inline bool isInitialized() { return g_state.initialized; }
inline const BackendState& state() { return g_state; }
inline OCLFuncs& func() { return g_func; }

// ─── GPU Memory Operations ───────────────────────────────────────

inline cl_mem gpuMalloc(size_t sizeBytes) {
    cl_int err;
    cl_mem buf = g_func.CreateBuffer(g_state.context, CL_MEM_READ_WRITE, sizeBytes, nullptr, &err);
    return (err == CL_SUCCESS) ? buf : nullptr;
}

inline void gpuFree(cl_mem buf) {
    if (buf) g_func.ReleaseMemObject(buf);
}

inline bool gpuWrite(cl_mem buf, const void* hostPtr, size_t sizeBytes) {
    return g_func.EnqueueWriteBuffer(g_state.queue, buf, 1, 0, sizeBytes, hostPtr, 0, nullptr, nullptr) == CL_SUCCESS;
}

inline bool gpuRead(cl_mem buf, void* hostPtr, size_t sizeBytes) {
    return g_func.EnqueueReadBuffer(g_state.queue, buf, 1, 0, sizeBytes, hostPtr, 0, nullptr, nullptr) == CL_SUCCESS;
}

inline bool gpuSync() {
    return g_func.Finish(g_state.queue) == CL_SUCCESS;
}

// ─── Kernel Compilation & Dispatch ──────────────────────────────

inline cl_program gpuCompileProgram(const char* source, size_t sourceLen) {
    cl_int err;
    const char* srcArr[1] = { source };
    const size_t lenArr[1] = { sourceLen };
    cl_program prog = g_func.CreateProgramWithSource(g_state.context, 1, srcArr, lenArr, &err);
    if (err != CL_SUCCESS || !prog) return nullptr;

    err = g_func.BuildProgram(prog, 1, &g_state.device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[4096] = {};
        g_func.GetProgramBuildInfo(prog, g_state.device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        g_func.ReleaseProgram(prog);
        return nullptr;
    }
    return prog;
}

inline cl_kernel gpuCreateKernel(cl_program prog, const char* funcName) {
    cl_int err;
    cl_kernel k = g_func.CreateKernel(prog, funcName, &err);
    return (err == CL_SUCCESS) ? k : nullptr;
}

inline bool gpuSetArg(cl_kernel kernel, cl_uint index, cl_size_t size, const void* value) {
    return g_func.SetKernelArg(kernel, index, size, value) == CL_SUCCESS;
}

inline bool gpuLaunch(cl_kernel kernel, size_t globalSize, size_t localSize) {
    cl_int err = g_func.EnqueueNDRangeKernel(
        g_state.queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr
    );
    return err == CL_SUCCESS;
}

inline void gpuReleaseKernel(cl_kernel k) { if (k) g_func.ReleaseKernel(k); }
inline void gpuReleaseProgram(cl_program p) { if (p) g_func.ReleaseProgram(p); }

inline void shutdown() {
    if (g_state.queue) { g_func.ReleaseCommandQueue(g_state.queue); g_state.queue = nullptr; }
    if (g_state.context) { g_func.ReleaseContext(g_state.context); g_state.context = nullptr; }
    g_state.initialized = false;
}

} // namespace xoc

#endif // XECUDA_OCL_H
