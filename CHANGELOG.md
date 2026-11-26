# Changelog

## 1.0.0 - 2026-11-26

First release.

- `maya-ik2` for HIP on Windows, built for `gfx1100`, `gfx1101` and `gfx1151`, with the weight formats of
  `Inkling-2-Mini-UD-IQ4_XS.gguf`.
- Matrix-core (WMMA) kernels for the prompt path: grouped experts, dense products, attention, loop gates.
- Single-token path with batched loads in its small kernels, matrix-core decode attention, and gate/up and down experts
  in one dispatch.
- Warm-up at load, a 4,096-token prompt chunk in the scripts, and an optional lane-per-query prompt attention
  (`IK2_FLASH2=4`).
- `build.ps1`, `run.ps1`, `serve.ps1`, the correctness gate and the benchmark script.

RX 7900 XT: 109 tok/s decode, 3,800 tok/s prompt at 4K tokens. Radeon 8060S: 47 tok/s decode, 1,485 tok/s prompt at 4K
tokens. Details in BENCHMARKS.md.
