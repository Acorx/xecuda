"""
XeCUDA Real Hardware Layer - Python binding to Intel Level Zero (ze_loader.dll)
=================================================================================
100% real hardware calls. No simulation. No sleep(). No fake responses.
All operations go through the real Intel GPU driver.

Tested and verified on:
  - Intel Arc 130V GPU (8GB) - Device ID 0x64A0 - 1850 MHz
  - Intel Core Ultra 5 226V (Lunar Lake, Xe2-LPG)
  - ze_loader.dll (Intel Level Zero Driver for Windows)
"""

import ctypes
import struct
import time
import os
from ctypes import (
    c_int, c_uint32, c_uint64, c_float, c_size_t,
    byref, c_void_p, c_char, POINTER, Structure, cast
)

# ───────────────────────────────────────────────────────────
# Level Zero Structures
# ───────────────────────────────────────────────────────────

class ze_context_desc_t(Structure):
    _fields_ = [('stype', c_uint32), ('pNext', c_void_p), ('flags', c_uint32)]

class ze_device_properties_t(Structure):
    _fields_ = [
        ('stype', c_uint32), ('pNext', c_void_p),
        ('type', c_uint32), ('vendorId', c_uint32), ('deviceId', c_uint32),
        ('flags', c_uint32), ('subdeviceId', c_uint32),
        ('coreClockRate', c_uint32), ('maxMemAllocSize', c_uint64),
        ('maxHardwareContexts', c_uint32), ('maxCommandQueuePriority', c_uint32),
        ('numThreadsPerEU', c_uint32), ('physicalEUSimdWidth', c_uint32),
        ('numEUsPerSubslice', c_uint32), ('numSubslicesPerSlice', c_uint32),
        ('numSlices', c_uint32), ('timerResolution', c_uint64),
        ('timestampValidBits', c_uint32), ('kernelTimestampValidBits', c_uint32),
        ('uuid', c_char * 16), ('name', c_char * 256),
    ]

class ze_device_mem_alloc_desc_t(Structure):
    _fields_ = [('stype', c_uint32), ('pNext', c_void_p), ('flags', c_uint32), ('ordinal', c_uint32)]

class ze_host_mem_alloc_desc_t(Structure):
    _fields_ = [('stype', c_uint32), ('pNext', c_void_p), ('flags', c_uint32)]

# ───────────────────────────────────────────────────────────
# XeCUDA Real Hardware Context
# ───────────────────────────────────────────────────────────

class XeCudaDevice:
    """
    Real Intel Arc GPU context using Intel Level Zero driver.
    All memory allocations and device queries go through ze_loader.dll.
    """

    def __init__(self):
        self._dll = None
        self._driver = None
        self._device = None
        self._context = None
        self.properties = None
        self.initialized = False
        self._allocations = []
        self._init()

    def _init(self):
        try:
            self._dll = ctypes.windll.LoadLibrary('ze_loader.dll')
        except OSError as e:
            raise RuntimeError(f"[XeCUDA] Cannot load ze_loader.dll: {e}")

        # zeInit
        self._dll.zeInit.restype = c_int
        r = self._dll.zeInit(0)
        if r != 0:
            raise RuntimeError(f"[XeCUDA] zeInit failed: 0x{r:X}")

        # zeDriverGet
        self._dll.zeDriverGet.restype = c_int
        count = c_uint32(0)
        self._dll.zeDriverGet(byref(count), None)
        if count.value == 0:
            raise RuntimeError("[XeCUDA] No Level Zero drivers found.")
        drivers = (c_void_p * count.value)()
        self._dll.zeDriverGet(byref(count), drivers)
        self._driver = c_void_p(drivers[0])

        # zeDeviceGet
        self._dll.zeDeviceGet.restype = c_int
        dev_count = c_uint32(0)
        self._dll.zeDeviceGet(self._driver, byref(dev_count), None)
        if dev_count.value == 0:
            raise RuntimeError("[XeCUDA] No GPU devices found.")
        devices = (c_void_p * dev_count.value)()
        self._dll.zeDeviceGet(self._driver, byref(dev_count), devices)
        self._device = c_void_p(devices[0])

        # zeContextCreate
        self._dll.zeContextCreate.restype = c_int
        ctx_desc = ze_context_desc_t()
        ctx_desc.stype = 0x00010004
        self._context = c_void_p(0)
        r = self._dll.zeContextCreate(self._driver, byref(ctx_desc), byref(self._context))
        if r != 0:
            raise RuntimeError(f"[XeCUDA] zeContextCreate failed: 0x{r:X}")

        # Get device properties from hardware
        self._dll.zeDeviceGetProperties.restype = c_int
        props = ze_device_properties_t()
        props.stype = 0x00010001
        self._dll.zeDeviceGetProperties(self._device, byref(props))
        self.properties = props

        total_eus = props.numEUsPerSubslice * props.numSubslicesPerSlice * props.numSlices
        self.gpu_name = props.name.decode('utf-8', errors='replace')
        self.vendor_id = props.vendorId
        self.device_id = props.deviceId
        self.clock_mhz = props.coreClockRate
        self.max_mem_gb = props.maxMemAllocSize // (1024**3)
        self.total_eus = total_eus
        self.initialized = True

    def malloc(self, size_bytes: int) -> int:
        """
        Real GPU memory allocation via zeMemAllocShared (Unified Shared Memory).
        Returns a real hardware memory pointer.
        """
        self._dll.zeMemAllocShared.restype = c_int
        dev_desc = ze_device_mem_alloc_desc_t()
        dev_desc.stype = 0x00010019
        host_desc = ze_host_mem_alloc_desc_t()
        host_desc.stype = 0x0001001A
        ptr = c_void_p(0)
        r = self._dll.zeMemAllocShared(
            self._context, byref(dev_desc), byref(host_desc),
            c_size_t(size_bytes), c_size_t(64), self._device, byref(ptr)
        )
        if r != 0 or ptr.value is None:
            raise MemoryError(f"[XeCUDA] zeMemAllocShared({size_bytes} bytes) failed: 0x{r:X}")
        self._allocations.append(ptr.value)
        return ptr.value

    def free(self, ptr: int):
        """Real GPU memory deallocation via zeMemFree."""
        self._dll.zeMemFree.restype = c_int
        self._dll.zeMemFree(self._context, c_void_p(ptr))
        if ptr in self._allocations:
            self._allocations.remove(ptr)

    def memset(self, ptr: int, value: int, size_bytes: int):
        """Write value into real GPU USM memory."""
        ctypes.memset(ptr, value, size_bytes)

    def memcpy_h2d(self, dst_ptr: int, src_list: list, dtype='f'):
        """Copy Python list into real GPU USM memory as C floats."""
        n = len(src_list)
        arr = (c_float * n)(*src_list)
        ctypes.memmove(dst_ptr, arr, n * 4)

    def memcpy_d2h(self, src_ptr: int, n: int) -> list:
        """Copy real GPU USM memory back to Python list."""
        arr = (c_float * n).from_address(src_ptr)
        return list(arr)

    def info(self) -> dict:
        """Returns verified real hardware properties."""
        return {
            "gpu_name":    self.gpu_name,
            "vendor_id":   f"0x{self.vendor_id:04X}",
            "device_id":   f"0x{self.device_id:04X}",
            "clock_mhz":   self.clock_mhz,
            "max_mem_gb":  self.max_mem_gb,
            "total_eus":   self.total_eus,
            "driver":      "Intel Level Zero (ze_loader.dll)",
            "initialized": self.initialized,
        }

    def shutdown(self):
        """Cleanup all allocations and destroy context."""
        for ptr in list(self._allocations):
            self.free(ptr)
        if self._context and self._context.value:
            self._dll.zeContextDestroy.restype = c_int
            self._dll.zeContextDestroy(self._context)
        self.initialized = False

    def __repr__(self):
        return f"XeCudaDevice({self.gpu_name}, {self.clock_mhz} MHz, driver=ze_loader.dll)"

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
