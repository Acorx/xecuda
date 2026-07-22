"""
Example 03: LLM & AI Inference Acceleration on Intel Arc 130V
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from xecuda import IntelArcInferenceEngine

print("=== XeCUDA Example 03: AI & LLM Inference Acceleration ===")

# Create inference engine targeting Intel Arc 130V XMX matrix engines
engine = IntelArcInferenceEngine(model_name="Llama-3-8B-Instruct", precision="fp16")
engine.load_model()

prompt = "Explain how Intel Arc Xe2 XMX acceleration works for neural networks."
result = engine.generate(prompt=prompt, max_new_tokens=120)

print("\n--- Model Output ---")
print(result)
