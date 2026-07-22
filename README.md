# XeCUDA 🚀
### Open-Source CUDA Compatibility Layer & AI Acceleration Suite for Intel Arc GPUs

**XeCUDA** is an open-source high-performance C++/Python library that brings NVIDIA CUDA API compatibility and AI compute acceleration directly to **Intel Arc GPUs**, with specialized optimizations for **Intel Arc 130V / 140V (Lunar Lake Xe2 Architecture)**.

---

## 💻 Targeted Hardware Specifications (Detected System)

| Feature | Specification |
| :--- | :--- |
| **GPU** | **Intel(R) Arc(TM) 130V GPU (8GB Shared VRAM)** |
| **GPU Architecture** | **Xe2-LPG (Battlemage Generation)** |
| **Xe Cores** | **7 Xe2 Cores** @ 1850 MHz |
| **Device ID / Vendor ID** | Device `0x64A0` \| Vendor `0x8086` (Subsystem `184E:1025`) |
| **CPU** | **Intel(R) Core(TM) Ultra 5 226V** (8 Cores / 8 Threads @ 2.10 GHz) |
| **System RAM** | **16 GB LPDDR5x @ 8533 MT/s** (High-Bandwidth Shared Architecture) |
| **Hardware Accelerators** | **Intel Xe Matrix Extensions (XMX)** & Ray Tracing Cores |

---

## ✨ Features

- **CUDA Runtime API Emulation (`xecuda_runtime.h`)**: Drop-in replacements for `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaStreamSynchronize`, `cudaGetDeviceProperties`, and `cudaGetLastError`.
- **CUDA-Style Kernel Syntax (`xecuda_kernel.h`)**: Supports standard CUDA kernel modifiers (`__global__`, `__device__`, `__shared__`) and thread indexing (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`).
- **Intel Arc XMX Matrix Engine Acceleration (`xecuda_blas.h`)**: High-throughput matrix multiplication (`xeCublasSgemm`, `xeCublasHgemmXMX`) leveraging Xe2 systolic matrix pipelines.
- **Zero-Code PyTorch Migration (`xecuda.torch_bridge`)**: Transparently maps `torch.cuda.*` calls to Intel's native `torch.xpu` SYCL engine.
- **High-Performance AI & LLM Inference (`xecuda.inference`)**: Optimized inference engine for Llama 3, DeepSeek, Qwen, and Stable Diffusion models on Intel Arc 130V.

---

## 📊 CUDA API vs XeCUDA Mapping Table

| NVIDIA CUDA API | XeCUDA Equivalent | Functionality on Intel Arc |
| :--- | :--- | :--- |
| `cudaMalloc(&ptr, size)` | `xeCudaMalloc(&ptr, size)` | 64-byte aligned allocation on Intel Shared VRAM |
| `cudaFree(ptr)` | `xeCudaFree(ptr)` | Releases GPU memory |
| `cudaMemcpy(dst, src, n, kind)` | `xeCudaMemcpy(dst, src, n, kind)` | High-bandwidth zero-copy memory transfer |
| `kernel<<<grid, block>>>(...)` | `XECUDA_LAUNCH(kernel, grid, block, ...)` | Launches SIMD parallel kernel on Xe Cores |
| `threadIdx`, `blockIdx` | `threadIdx`, `blockIdx` | Thread & block dimension indexing |
| `cublasSgemm(...)` | `xeCublasSgemm(...)` | Matrix multiplication on Intel Xe2 XMX |
| `torch.cuda.is_available()` | `xecuda.init_torch_cuda_bridge()` | PyTorch CUDA to Intel XPU dynamic redirection |

---

## 🛠️ Building & Running (C++)

### Prerequisites
- C++17 compatible compiler (MSVC on Windows, GCC, or Clang)
- CMake 3.15+

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/XeCUDA.git
cd XeCUDA

# Create build directory
mkdir build && cd build

# Configure & build with CMake
cmake ..
cmake --build . --config Release

# Run the XeCUDA Diagnostic & Benchmark CLI
./Release/xecuda-bench.exe
```

---

## 🐍 Python Usage & AI Inferences

### 1. PyTorch CUDA Auto-Bridge
```python
import xecuda

# Transparently hook torch.cuda calls to Intel Arc GPU (torch.xpu)
xecuda.init_torch_cuda_bridge()

import torch

# This now executes natively on Intel Arc 130V!
x = torch.randn(1024, 1024, device="cuda")
y = torch.matmul(x, x)
print("Result computed on:", torch.cuda.get_device_name(0))
```

### 2. Fast LLM Inferences on Intel Arc 130V
```python
from xecuda import IntelArcInferenceEngine

# Load LLM with Xe2 XMX matrix engine acceleration
engine = IntelArcInferenceEngine(model_name="Llama-3-8B-Instruct", precision="fp16")
engine.load_model()

prompt = "Explain quantum computing in 2 sentences."
response = engine.generate(prompt=prompt, max_new_tokens=100)
print(response)
```

---

## 📜 License
Open-Source under the MIT License.
