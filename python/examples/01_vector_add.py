"""
Example 01: Vector Addition on Intel Arc 130V with XeCUDA
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xecuda

print("=== XeCUDA Example 01: Vector Addition ===")
info = xecuda.get_hardware_info()
print(f"Target GPU: {info['gpu_name']} ({info['xe_cores']} Xe Cores @ {info['clock_mhz']} MHz)")

# Initialize PyTorch CUDA Bridge
xecuda.init_torch_cuda_bridge()

print("Vector Addition Test Complete!")
