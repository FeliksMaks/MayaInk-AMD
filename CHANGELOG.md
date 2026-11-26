# Changelog

## 1.2.0 - 2026-11-26

- Radeon 8060S: the prompt attention computes its relative-position bias itself instead of reading a precomputed table.
  Prompt reading 1,426 -> 1,621 tok/s at 4K tokens, 1,236 -> 1,368 at 32K. Same tokens as before. The RX 7900 XT keeps the
  table (faster there). `IK2_FLASH_INL1=0` / `1` forces either.

## 1.1.0 - 2026-11-26

- FP8 KV cache: single-token attention runs on the matrix cores (WMMA), as with the FP16 cache. Decode with `-Fp8Kv` on an
  RX 7900 XT: 94 -> 103 tok/s after 4K tokens, 90 -> 96 after 32K, 75 -> 81 after 95K. Radeon 8060S: 42 -> 44 and 38 -> 39.
  `IK2_ATTN_WMMA8=0` selects the earlier kernel. The FP16 cache is unchanged.

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
