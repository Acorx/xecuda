"""
XeCUDA Device Layer — Real Intel Arc GPU via OpenCL 3.0
======================================================
ALL memory operations go through OpenCL buffers on the actual Intel Arc 130V GPU.
ALL device queries come from the real hardware via clGetDeviceInfo.
Zero simulation. Zero fake values. Real GPU.
"""

import ctypes
import os

CL_DEVICE_TYPE_ALL = ctypes.c_ulonglong(0xFFFFFFFF)
CL_MEM_READ_WRITE = 0x0001
CL_SUCCESS = 0


def _P(v):
    return ctypes.c_void_p(v)


class XeCudaDevice:
    """
    Real Intel Arc GPU context via OpenCL 3.0 (Intel NEO driver).
    All memory is allocated as cl_mem GPU buffers.
    """

    def __init__(self):
        self._ocl = ctypes.WinDLL("OpenCL.dll")
        self._ocl.clCreateContext.restype = ctypes.c_void_p
        self._ocl.clCreateBuffer.restype = ctypes.c_void_p
        self._ocl.clCreateProgramWithSource.restype = ctypes.c_void_p
        self._ocl.clCreateKernel.restype = ctypes.c_void_p
        self._ocl.clCreateCommandQueueWithProperties.restype = ctypes.c_void_p

        self._programs = {}
        self._kernels = {}
        self._buffers = []
        self.initialized = False
        self._init()

    def _init(self):
        n_plat = ctypes.c_uint32(0)
        self._ocl.clGetPlatformIDs(0, None, ctypes.byref(n_plat))
        platforms = (ctypes.c_void_p * n_plat.value)()
        self._ocl.clGetPlatformIDs(n_plat, platforms, None)
        self._plat = _P(platforms[0])

        dc = ctypes.c_uint32(0)
        self._ocl.clGetDeviceIDs(self._plat, CL_DEVICE_TYPE_ALL, 0, None, ctypes.byref(dc))
        devs = (ctypes.c_void_p * dc.value)()
        self._ocl.clGetDeviceIDs(self._plat, CL_DEVICE_TYPE_ALL, dc, devs, None)
        self._dev = _P(devs[0])
        self._devs_arr = devs

        name_buf = ctypes.create_string_buffer(256)
        self._ocl.clGetDeviceInfo(self._dev, 0x102F, 256, name_buf, None)
        self.gpu_name = name_buf.value.decode()

        cu = ctypes.c_uint32(0)
        self._ocl.clGetDeviceInfo(self._dev, 0x1002, 4, ctypes.byref(cu), None)
        self.num_compute_units = cu.value

        vendor_buf = ctypes.create_string_buffer(256)
        self._ocl.clGetDeviceInfo(self._dev, 0x102C, 256, vendor_buf, None)
        self.vendor = vendor_buf.value.decode()

        # CL_DEVICE_VENDOR_ID = 0x1001 (real PCI vendor id, e.g. 0x8086 = Intel)
        vid = ctypes.c_uint32(0)
        self._ocl.clGetDeviceInfo(self._dev, 0x1001, 4, ctypes.byref(vid), None)
        self.vendor_id = hex(vid.value)

        self._ocl.clGetDeviceInfo(self._dev, 0x1097, 256, name_buf, None)
        self.driver_version = name_buf.value.decode()
        global_mem = ctypes.c_ulonglong(0)
        self._ocl.clGetDeviceInfo(self._dev, 0x101F, 8, ctypes.byref(global_mem), None)
        self.total_memory_bytes = global_mem.value

        max_alloc = ctypes.c_ulonglong(0)
        self._ocl.clGetDeviceInfo(self._dev, 0x1010, 8, ctypes.byref(max_alloc), None)
        self.max_alloc_bytes = max_alloc.value

        err = ctypes.c_int32(0)
        self._ctx = _P(self._ocl.clCreateContext(None, 1, devs, None, None, ctypes.byref(err)))
        if err.value != 0:
            raise RuntimeError(f"[XeCUDA] clCreateContext failed: {err.value}")

        err3 = ctypes.c_int32(0)
        self._queue = _P(self._ocl.clCreateCommandQueueWithProperties(self._ctx, self._dev, None, ctypes.byref(err3)))
        if err3.value != 0:
            raise RuntimeError(f"[XeCUDA] clCreateCommandQueueWithProperties failed: {err3.value}")

        self.initialized = True

    def malloc(self, size_bytes: int) -> int:
        """Allocate GPU buffer via OpenCL. Returns cl_mem handle as int."""
        err = ctypes.c_int32(0)
        buf = self._ocl.clCreateBuffer(self._ctx, CL_MEM_READ_WRITE, size_bytes, None, ctypes.byref(err))
        if err.value != 0 or buf is None:
            raise MemoryError(f"[XeCUDA] clCreateBuffer({size_bytes} bytes) failed: err={err.value}")
        handle = buf if isinstance(buf, int) else ctypes.cast(buf, ctypes.c_void_p).value
        self._buffers.append(handle)
        return handle

    def free(self, buf_handle: int):
        """Release GPU buffer."""
        if buf_handle in self._buffers:
            self._buffers.remove(buf_handle)
        self._ocl.clReleaseMemObject(_P(buf_handle))

    def write_buffer(self, buf_handle: int, host_ptr, size_bytes: int):
        """Copy host data to GPU buffer via clEnqueueWriteBuffer."""
        self._ocl.clEnqueueWriteBuffer(
            self._queue, _P(buf_handle), 1, 0, size_bytes,
            host_ptr, 0, None, None
        )

    def read_buffer(self, buf_handle: int, host_ptr, size_bytes: int):
        """Copy GPU buffer to host data via clEnqueueReadBuffer."""
        self._ocl.clEnqueueReadBuffer(
            self._queue, _P(buf_handle), 1, 0, size_bytes,
            host_ptr, 0, None, None
        )

    def memcpy_h2d(self, dst_handle: int, src_list: list, dtype="f"):
        """Copy Python list into GPU buffer."""
        n = len(src_list)
        arr = (ctypes.c_float * n)(*src_list)
        self.write_buffer(dst_handle, ctypes.byref(arr), n * 4)

    def memcpy_d2h(self, src_handle: int, n: int) -> list:
        """Copy GPU buffer to Python list."""
        arr = (ctypes.c_float * n)()
        self.read_buffer(src_handle, ctypes.byref(arr), n * 4)
        return list(arr)

    def memset(self, buf_handle: int, value: int, size_bytes: int):
        """Fill GPU buffer with value using a simple kernel."""
        k = self.build_kernel("memset_fill", b"""
__kernel void memset_fill(__global uint* buf, const uint val, const int N) {
    int i = get_global_id(0);
    if (i < N) buf[i] = val;
}
""", b"memset_fill")
        fill_u32 = (value & 0xFF) * 0x01010101
        n_words = size_bytes // 4
        self.enqueue_kernel(k, (n_words + 255) // 256 * 256, 256, [
            ctypes.c_void_p(buf_handle),
            ctypes.c_int32(fill_u32),
            ctypes.c_int32(n_words),
        ])
        self.finish()

    def _get_program(self, name: str, source: bytes) -> ctypes.c_void_p:
        """Get or compile an OpenCL program (cached)."""
        if name in self._programs:
            return self._programs[name]
        sa = (ctypes.c_char_p * 1)(source)
        la = (ctypes.c_size_t * 1)(len(source))
        err = ctypes.c_int32(0)
        prog = _P(self._ocl.clCreateProgramWithSource(self._ctx, 1, sa, la, ctypes.byref(err)))
        if err.value != 0:
            raise RuntimeError(f"[XeCUDA] clCreateProgramWithSource failed: {err.value}")
        build_r = self._ocl.clBuildProgram(prog, 1, self._devs_arr, None, None, None)
        if build_r != 0:
            log = ctypes.create_string_buffer(4096)
            self._ocl.clGetProgramBuildInfo(prog, self._dev, 0x1084, 4096, log, None)
            raise RuntimeError(f"[XeCUDA] Kernel build failed: {log.value.decode()}")
        self._programs[name] = prog
        return prog

    def build_kernel(self, kernel_name: str, source: bytes, func_name: bytes) -> ctypes.c_void_p:
        """Compile and return an OpenCL kernel (cached)."""
        key = (kernel_name, func_name)
        if key in self._kernels:
            return self._kernels[key]
        prog = self._get_program(kernel_name, source)
        err = ctypes.c_int32(0)
        k = _P(self._ocl.clCreateKernel(prog, func_name, ctypes.byref(err)))
        if err.value != 0:
            raise RuntimeError(f"[XeCUDA] clCreateKernel('{func_name.decode()}') failed: {err.value}")
        self._kernels[key] = k
        return k

    def enqueue_kernel(self, kernel, global_size, local_size, args):
        """Launch an OpenCL kernel (1D) with the given arguments."""
        for i, arg in enumerate(args):
            if isinstance(arg, int):
                val = ctypes.c_void_p(arg)
                self._ocl.clSetKernelArg(kernel, i, ctypes.sizeof(val), ctypes.pointer(val))
            elif isinstance(arg, ctypes.c_int32):
                self._ocl.clSetKernelArg(kernel, i, 4, ctypes.pointer(arg))
            elif isinstance(arg, ctypes.c_float):
                self._ocl.clSetKernelArg(kernel, i, 4, ctypes.pointer(arg))
            elif isinstance(arg, ctypes.c_void_p):
                self._ocl.clSetKernelArg(kernel, i, ctypes.sizeof(arg), ctypes.pointer(arg))
            else:
                raise TypeError(f"Unsupported arg type: {type(arg)}")
        gs = ctypes.c_size_t(global_size)
        ls = ctypes.c_size_t(local_size)
        self._ocl.clEnqueueNDRangeKernel(
            self._queue, kernel, 1, None,
            ctypes.pointer(gs), ctypes.pointer(ls), 0, None, None
        )

    def enqueue_kernel_2d(self, kernel, global_sizes, local_sizes, args):
        """Launch an OpenCL kernel (2D) with the given arguments."""
        for i, arg in enumerate(args):
            if isinstance(arg, int):
                val = ctypes.c_void_p(arg)
                self._ocl.clSetKernelArg(kernel, i, ctypes.sizeof(val), ctypes.pointer(val))
            elif isinstance(arg, ctypes.c_int32):
                self._ocl.clSetKernelArg(kernel, i, 4, ctypes.pointer(arg))
            elif isinstance(arg, ctypes.c_float):
                self._ocl.clSetKernelArg(kernel, i, 4, ctypes.pointer(arg))
            elif isinstance(arg, ctypes.c_void_p):
                self._ocl.clSetKernelArg(kernel, i, ctypes.sizeof(arg), ctypes.pointer(arg))
            else:
                raise TypeError(f"Unsupported arg type: {type(arg)}")
        gs = (ctypes.c_size_t * 2)(*global_sizes)
        ls = (ctypes.c_size_t * 2)(*local_sizes)
        self._ocl.clEnqueueNDRangeKernel(
            self._queue, kernel, 2, None,
            ctypes.cast(gs, ctypes.POINTER(ctypes.c_size_t)),
            ctypes.cast(ls, ctypes.POINTER(ctypes.c_size_t)), 0, None, None
        )

    def finish(self):
        """Synchronize: wait for all GPU operations to complete."""
        self._ocl.clFinish(self._queue)

    def info(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "vendor": self.vendor,
            "driver": self.driver_version,
            "compute_units": self.num_compute_units,
            "vendor_id": getattr(self, "vendor_id", "0x0"),
            "total_memory_gb": round(self.total_memory_bytes / (1024**3), 2),
            "max_alloc_gb": round(self.max_alloc_bytes / (1024**3), 2),
            "initialized": self.initialized,
            "backend": "OpenCL 3.0 Intel NEO (Real GPU)",
        }

    def shutdown(self):
        for h in list(self._buffers):
            try:
                self.free(h)
            except Exception:
                pass
        if hasattr(self, "_queue") and self._queue:
            self._ocl.clReleaseCommandQueue(self._queue)
        if hasattr(self, "_ctx") and self._ctx:
            self._ocl.clReleaseContext(self._ctx)
        self.initialized = False

    def __repr__(self):
        return f"XeCudaDevice({self.gpu_name}, {self.num_compute_units} CU, OpenCL 3.0)"

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
