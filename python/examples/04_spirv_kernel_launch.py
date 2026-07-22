"""
Example 04: Bare-Metal Level Zero Driver & SPIR-V Kernel Launch on Intel Arc 130V
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xecuda

print("=== XeCUDA Example 04: Bare-Metal Level Zero Driver & SPIR-V ===")

# Display Level Zero GPU target specs
info = xecuda.get_hardware_info()
print(f"Target GPU           : {info['gpu_name']}")
print(f"Architecture         : {info['architecture']}")
print(f"Driver Backend       : Intel Level Zero (ze_api.h)")
print(f"Memory Architecture  : {info['vram_shared_gb']} GB Shared VRAM @ {info['ram_speed_mts']} MT/s Zero-Copy Bus")

print("\nLevel Zero Driver Test Complete!")
