"""
Example 07: Execution of Qwythos-9B-Claude-Mythos-5-1M GGUF Model on Intel Arc 130V
Target Path: C:\\Users\\arthu\\.lmstudio\\models\\empero-ai\\Qwythos-9B-Claude-Mythos-5-1M-GGUF\\Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xecuda
from xecuda.model_runner import XeCudaModelRunner

print("=== XeCUDA Example 07: Real 9B GGUF Model Execution on Intel Arc 130V ===")

model_path = r"C:\Users\arthu\.lmstudio\models\empero-ai\Qwythos-9B-Claude-Mythos-5-1M-GGUF\Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"

# Verify hardware specs
info = xecuda.get_hardware_info()
print(f"Target GPU   : {info['gpu_name']} ({info['xe_cores']} Xe Cores @ {info['clock_mhz']} MHz)")
print(f"Driver       : Intel Level Zero (ze_loader.dll)")
print(f"Memory Bus   : {info['vram_shared_gb']} GB Shared VRAM @ {info['ram_speed_mts']} MT/s LPDDR5x")

# Initialize runner for Qwythos 9B GGUF
runner = XeCudaModelRunner(model_path)
runner.load_to_vram()

# Run inference test
prompt = "Explain your architectural advantages running on Intel Arc 130V GPU with XeCUDA."
runner.generate_response(prompt=prompt, max_tokens=150)

print("\nModel Execution Complete!")
