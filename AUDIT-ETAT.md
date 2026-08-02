# Audit technique & état réel — XeCUDA

État vérifié et mesuré sur la machine cible (Intel Core Ultra, **Arc 130V** via
OpenCL 3.0 Intel NEO) à la date du 2026-08-03. Objectif : honnêteté totale sur
ce qui est **réel / mesuré** vs **émulé / déclaratif**.

## Où en est le projet

| Brique | Réel ? | Statut mesuré |
| :--- | :--- | :--- |
| `device.py` (XeCudaDevice) — backend **OpenCL** pur `ctypes`, zéro dépendance | ✅ Réel GPU | Init JA sur l'Arc (vendor `0x8086`, 7.96 GB, 56 EU, driver OpenCL 3.0 NEO) |
| Kernels du GPU : add, SGEMM, matvec_f32, rms_norm, softmax, **GQA attention**, SiLU | ✅ Réel GPU (mono launch) | **8/8 PASS** corrects vs référence sur l'Arc réel |
| Torch CUDA→XPU bridge | ⚠️ Honnête mais **no-op actuellement** | `torch.xpu` indisponible (wheels Py 3.1x/3.14, IPEX bloqué) → `is_available()=False` et le bridge se désactive proprement |
| `xecuda_xmx.cpp` « XMX » (C++) | ❌ **Émulation CPU OpenMP** | boucle scaire triple sur `float`, pas d'instructions XMX/systoliques |
| `xecuda_flash_attn.h` « FlashAttention-2 » | ❌ **Émulation CPU** | softmax naïve `O(n²·d)`, aucun tile mem / tiles L2 ni online-max réel |
| `xecuda_l0.cpp` « Level Zero » | ❌ **Alias OpenCL** | handles L0 = fake (1..5), réutilise le backend OpenCL réel |
| README claims (LLM Llama/DeepSeek/Qwen/StableDiff sur Arc) | 🟠 aspirant | l'inférence nécessite un GGUF réel + llama-cpp-python ; pas encore e2e |
| `benchmark_bandwidth` | ✅ mesuré | write ≈ 2.4 GB/s, read ≈ 9.2 GB/s (via clEnqueueWriteBuffer, iGPU) |

## Cleanup fait (2026-08-03)
- `__init__.py` rendu **importable sans numpy** (GPU core = ctypes pur) → corrige le venv/numpy cassé.
- kernels GPU corrigés : **SGEMM** (mapping row-major exact, précedent 16-col bug), `vector_add`, alias `matvec_q4km`.
- `device.info()` lit le **`vendor_id` réel** (CL_DEVICE_VENDOR_ID).
- Suite de validation `tests/validate_real_gpu.py` : preuve que **les kernels tournent correctement sur le GPU réel** (pas de simulation).

## Prochaines priorités honnêtes
1. **SGEMM performant** : tiling bloc 8x8/shared-mem + clEnqueueWrite optimisé (write ~2.4GB/s est très bas), ou passer en BYZ/gguf native.
2. **Vraie inference** : brancher un GGUF réel (ex. Qwythos-9B) via `matvec_q4k/q6k` + layer transformer, mesurer tok/s réel.
3. Rebrander/remocler les noms trompeurs (XMX, FlashAttention-2, Level Zero) vers leur réalité (OpenCL).

## Environnement build C++
- cmake ✓ (`C:\Program Files\CMake`), ninja ✓, clang-cl ✓ (LLVM).
- Pas de MSVC/Windows SDK → clang-cl ne compile pas les headers Win. MSYS2/gcc en cours d'installation.