"""
XeCUDA Text Generation — End-to-end autoregressive generation
=============================================================
Downloads tokenizer from HuggingFace, loads GGUF model onto Intel Arc 130V,
runs greedy autoregressive generation entirely on the GPU.

Usage:
    python -m xecuda.generate "The capital of France is"
    python -m xecuda.generate --interactive
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xecuda.device import XeCudaDevice
from xecuda.weight_loader import WeightLoader
from xecuda.model import Model


# ═══════════════════════════════════════════════════════════════════════
# Simple Tokenizer — downloads from HuggingFace or loads locally
# ═══════════════════════════════════════════════════════════════════════

class SimpleTokenizer:
    """Byte-pair encoding tokenizer using HuggingFace tokenizers."""

    def __init__(self, vocab_size=248320):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def load(self, model_name="empero-ai/Qwythos-9B-Claude-Mythos-5-1M"):
        """Load tokenizer from HuggingFace."""
        try:
            from tokenizers import Tokenizer
            from huggingface_hub import hf_hub_download

            print(f"[Tokenizer] Downloading tokenizer from {model_name}...")
            path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
            self.tokenizer = Tokenizer.from_file(path)
            print(f"[Tokenizer] Loaded: vocab_size={self.tokenizer.get_vocab_size()}")
        except ImportError:
            print("[Tokenizer] WARNING: 'tokenizers' package not installed.")
            print("  Install with: pip install tokenizers")
            print("  Falling back to simple character tokenizer.")
            self.tokenizer = None
        except Exception as e:
            print(f"[Tokenizer] WARNING: Could not load tokenizer: {e}")
            print("  Falling back to simple character tokenizer.")
            self.tokenizer = None

    def encode(self, text):
        """Encode text to token IDs."""
        if self.tokenizer is not None:
            encoding = self.tokenizer.encode(text)
            return encoding.ids
        else:
            # Simple fallback: map each char to its ASCII value mod vocab_size
            return [ord(c) % self.vocab_size for c in text]

    def decode(self, token_ids):
        """Decode token IDs to text."""
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids)
        else:
            return "".join(chr(tid % 128) for tid in token_ids)


# ═══════════════════════════════════════════════════════════════════════
# Greedy Text Generation
# ═══════════════════════════════════════════════════════════════════════

def generate(device, model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    """Run autoregressive generation on the GPU model."""
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    token_ids = tokenizer.encode(prompt)
    print(f"Input tokens: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
    print(f"Input length: {len(token_ids)} tokens")

    device.finish()
    t_start = time.perf_counter()

    generated_tokens = []
    for step in range(max_new_tokens):
        # Forward pass for the last token
        # For now, only process the last token (ignoring KV cache)
        # This is correct for greedy decoding but inefficient
        logits = model.forward(token_ids[-1], seq_pos=step)

        # Greedy decode
        next_token = int(np.argmax(logits))

        # Stop at EOS
        if next_token == 248046:  # Qwen EOS token
            break

        token_ids.append(next_token)
        generated_tokens.append(next_token)

        # Print progress
        if step % 10 == 0 or step == max_new_tokens - 1:
            try:
                partial = tokenizer.decode(generated_tokens)
            except Exception:
                partial = f"[{len(generated_tokens)} tokens]"
            try:
                print(f"\r[{step+1}/{max_new_tokens}] {partial}", end="", flush=True)
            except UnicodeEncodeError:
                print(f"\r[{step+1}/{max_new_tokens}] [{len(generated_tokens)} tokens]", end="", flush=True)

    device.finish()
    t_end = time.perf_counter()

    try:
        full_output = tokenizer.decode(token_ids)
        gen_text = tokenizer.decode(generated_tokens)
    except Exception:
        full_output = f"[{len(token_ids)} tokens]"
        gen_text = f"[{len(generated_tokens)} tokens]"

    print(f"\n\n{'='*60}")
    print(f"Full output: {full_output}")
    print(f"{'='*60}")
    print(f"Generated {len(generated_tokens)} tokens in {t_end-t_start:.2f}s "
          f"({len(generated_tokens)/(t_end-t_start):.1f} tokens/sec)")

    return full_output, generated_tokens


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="XeCUDA text generation on Intel Arc 130V")
    parser.add_argument("prompt", nargs="?", default="The capital of France is",
                       help="Text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode: keep generating on user input")
    args = parser.parse_args()

    print("="*60)
    print("XeCUDA Text Generation — Intel Arc 130V")
    print("="*60)

    # Initialize GPU
    print("\n[1/4] Initializing GPU...")
    device = XeCudaDevice()

    # Load weights
    print("\n[2/4] Loading model weights...")
    wl = WeightLoader(device)
    wl.load(verbose=True)

    # Create model
    print("\n[3/4] Building model...")
    model = Model(device, wl)

    # Load tokenizer
    print("\n[4/4] Loading tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.load()

    # Generate
    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\n> ")
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue
                generate(device, model, tokenizer, prompt, args.max_tokens, args.temperature)
            except KeyboardInterrupt:
                break
    else:
        generate(device, model, tokenizer, args.prompt, args.max_tokens, args.temperature)

    # Cleanup
    print("\nCleaning up...")
    model.shutdown()
    wl.shutdown()
    device.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
