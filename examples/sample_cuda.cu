// Sample CUDA Source File (.cu)
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void myCudaKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

int main() {
    int n = 1024;
    size_t bytes = n * sizeof(float);

    float* d_data;
    cudaMalloc((void**)&d_data, bytes);

    myCudaKernel<<<1, 1024>>>(d_data, n);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    printf("CUDA Kernel executed successfully on Intel Arc!\n");
    return 0;
}
