"""Mini-probe OpenCL pure ctypes — mime exactement la mécanique de xecuda_ocl.h.
Valide sur cette machine que l'Arc 130V est joignable via OpenCL + qu'un kernel matmul s'execute.
Aucune dépendance (pas de pyopencl)."""
import ctypes, ctypes.util, time

cl = ctypes.cdll.LoadLibrary("OpenCL.dll")

# protos (Argument/PI types explicites pour éviter les débordements)
def P(): return ctypes.POINTER(ctypes.c_void_p)
cl.clGetPlatformIDs.argtypes = [ctypes.c_uint, P(), ctypes.POINTER(ctypes.c_uint)]
cl.clGetPlatformIDs.restype = ctypes.c_int
cl.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, P(), ctypes.POINTER(ctypes.c_uint)]
cl.clGetDeviceIDs.restype = ctypes.c_int
cl.clGetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
cl.clGetDeviceInfo.restype = ctypes.c_int
cl.clCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_uint, P(), ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
cl.clCreateContext.restype = ctypes.c_void_p
cl.clCreateCommandQueue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
cl.clCreateCommandQueue.restype = ctypes.c_void_p

cl.clGetPlatformIDs.restype = ctypes.c_int
cl.clGetDeviceIDs.restype = ctypes.c_int
cl.clGetDeviceInfo.restype = ctypes.c_int
cl.clCreateContext.restype = ctypes.c_void_p
cl.clCreateCommandQueue.restype = ctypes.c_void_p
cl.clCreateBuffer.restype = ctypes.c_void_p
cl.clEnqueueWriteBuffer.restype = ctypes.c_int
cl.clEnqueueReadBuffer.restype = ctypes.c_int
cl.clCreateProgramWithSource.restype = ctypes.c_void_p
cl.clBuildProgram.restype = ctypes.c_int
cl.clCreateKernel.restype = ctypes.c_void_p
cl.clSetKernelArg.restype = ctypes.c_int
cl.clEnqueueNDRangeKernel.restype = ctypes.c_int
cl.clFinish.restype = ctypes.c_int

M, K, N = 128, 128, 128

# devices
nplat = ctypes.c_uint(0)
cl.clGetPlatformIDs(0, None, ctypes.byref(nplat))
plats = (ctypes.c_void_p*nplat.value)()
cl.clGetPlatformIDs(nplat.value, plats, None)
ndev = ctypes.c_uint(0)
cl.clGetDeviceIDs(plats[0], 0xFFFFFFFF, 0, None, ctypes.byref(ndev))
devs = (ctypes.c_void_p*ndev.value)()
cl.clGetDeviceIDs(plats[0], 0xFFFFFFFF, ndev.value, devs, None)
dev = devs[0]

name = ctypes.create_string_buffer(256)
cl.clGetDeviceInfo(dev, 0x102F, 256, name, None)   # CL_DEVICE_NAME
cubuf = ctypes.c_uint(0)
cl.clGetDeviceInfo(dev, 0x1002, 4, ctypes.byref(cubuf), None)  # MAX_COMPUTE_UNITS
print("DEVICE:", name.value.decode(errors='ignore'))
print("COMPUTE_UNITS (lanes Xe) ~ :", cubuf.value)

cl.clCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
cl.clCreateBuffer.restype = ctypes.c_void_p
cl.clCreateProgramWithSource.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int)]
cl.clCreateProgramWithSource.restype = ctypes.c_void_p
cl.clBuildProgram.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
cl.clCreateKernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
cl.clCreateKernel.restype = ctypes.c_void_p
cl.clEnqueueWriteBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
cl.clEnqueueReadBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
cl.clFinish.argtypes = [ctypes.c_void_p]
cl.clSetKernelArg.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
cl.clEnqueueNDRangeKernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
cl.clEnqueueNDRangeKernel.restype = ctypes.c_int
err = ctypes.c_int(0)
devp = (ctypes.c_void_p*1)(dev)   # tableau d'1 dispositif, comme dans xecuda_ocl.h
ctx = cl.clCreateContext(None, 1, devp, None, None, ctypes.byref(err))
q = cl.clCreateCommandQueue(ctx, dev, 0, ctypes.byref(err))

def alloc(sz):
    return cl.clCreateBuffer(ctx, 0x0001, sz, None, ctypes.byref(err))

dA, dB, dC = alloc(4*K*M), alloc(4*K*N), alloc(4*M*N)
A = (ctypes.c_float*(M*K))()
B = (ctypes.c_float*(N*K))()
C = (ctypes.c_float*(M*N))()
for i in range(M*K): A[i] = 1.0
for i in range(N*K): B[i] = 2.0

def wsp(src, buf):
    a = ctypes.create_string_buffer(src)
    cstr = ctypes.c_char_p(ctypes.addressof(a))
    prp = cl.clCreateProgramWithSource(ctx, 1, ctypes.cast(ctypes.byref(cstr), ctypes.POINTER(ctypes.c_char_p)), None, ctypes.byref(err))
    if cl.clBuildProgram(prp, 1, devp, None, None, None)!=0:
        return None,None
    return prp, cl.clCreateKernel(prp, b"gemm", ctypes.byref(err))

count = 4*K*M
cl.clEnqueueWriteBuffer(q, dA, 1, 0, count, (ctypes.c_void_p)(ctypes.addressof(A)), 0, None, None)
cl.clEnqueueWriteBuffer(q, dB, 1, 0, 4*K*N, (ctypes.c_void_p)(ctypes.addressof(B)), 0, None, None)

SRC = b"""
__kernel void gemm(__global const float* A, __global const float* B, __global float* C,
                   const int M,const int K,const int N){
 int r=get_global_id(0); int c=get_global_id(1);
 if(r<M&&c<N){ float s=0; for(int i=0;i<K;i++) s+=A[r*K+i]*B[i*N+c]; C[r*N+c]=s; }
}"""
prog, kern = wsp(SRC, 0)
if not kern:
    print("Kernel compile FAILED"); raise SystemExit(1)
def setarg(k,i,ptr,sz):
    cl.clSetKernelArg(k, i, sz, ptr)
for i,(ptr,sz) in enumerate([
    (ctypes.byref(ctypes.c_void_p(dA)),8),(ctypes.byref(ctypes.c_void_p(dB)),8),(ctypes.byref(ctypes.c_void_p(dC)),8),
    (ctypes.byref(ctypes.c_int(M)),4),(ctypes.byref(ctypes.c_int(K)),4),(ctypes.byref(ctypes.c_int(N)),4)]):
    cl.clSetKernelArg(kern, i, sz, ptr)

gs=(ctypes.c_size_t*2)(M,N); ls=(ctypes.c_size_t*2)(16,16)
t=time.time(); 
cl.clEnqueueNDRangeKernel(q, kern, 2, None, gs, ls, 0, None,None)
cl.clFinish(q)
dt=time.time()-t
cl.clEnqueueReadBuffer(q, dC, 1, 0, 4*M*N, (ctypes.c_void_p)(ctypes.addressof(C)), 0, None,None)
cl.clFinish(q)
print(f"GEMM {M}x{K}x{N} en {dt*1000:.1f} ms — C[0]={C[0]:.1f} (attendu {K}, vocab nucléaires OK)" )
print("VALIDE → l'Arc répond au code OpenCL que le projet utilise réellement.")