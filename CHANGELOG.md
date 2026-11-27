# Changelog

## 1.4.0 - 2026-11-27

- ISTA-DASLab's `Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf` runs as well as Unsloth's UD-IQ4_XS file: its `inkling2_loop`
  metadata (the loop's layers and count, the sliding pattern per physical layer), its loop-gate and message-query tensor
  names, BF16 matrices, and the i-quant expert formats IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S with IQ4_NL or Q2_0
  down, on the single-token path and in the prompt's matrix-core expert tiles. The grids come from ggml's
  `ggml-common.h` (`third_party/ggml`, MIT).
- GSQ-RCO decode: the dense attention weights (Q4_K, Q5_K, Q6_K, BF16) are converted to Q8_0 at load (`IK2_REQ8`, on
  by default for this file), and the expert kernels have compiled versions for IQ3_XXS and IQ4_NL (`IK2_TOKTY`). RX 7900
  XT 65 -> 103 tok/s with a short prompt, Radeon 8060S 34 -> 43. The conversion changes the output slightly, with the
  same agreement with the llama.cpp reference within near ties; `IK2_REQ8=0` keeps the file's weights (77 tok/s on the
  RX 7900 XT). `scripts/reference-gsq.json` is the reference for that file.
- The UD-IQ4_XS file gives the same tokens and speed as 1.3.0.
- Tested on an RX 7800 XT 16 GB (`gfx1101`) with the GSQ file: 83 tok/s decode, 2,508 tok/s prompt at 4K. The UD
  file does not fit there with every expert in VRAM.

## 1.3.0 - 2026-11-26

- The looped layers' message gates compute their two down products and their mix in one kernel, reading the layer's
  input and message once instead of twice (`IK2_GATE_FUSE=16`, on in the scripts). Prompt reading, against 1.2.0 in the
  same alternating runs: RX 7900 XT 3,444 -> 3,556 tok/s at 4K tokens (+3.3%), 3,287 -> 3,348 at 32K (+1.9%); Radeon 8060S
  1,564 -> 1,672 at 4K (+6.9%), 1,268 -> 1,295 at 32K (+2.1%). Decode is unchanged. The sums run in another order, so
  some generated tokens differ from 1.2.0; the agreement with the llama.cpp reference is the same (188/188 and 192/192
  short, 93/94 and 95/96 long, on both GPUs). `-NoGateFuse` in `run.ps1` and `serve.ps1` gives 1.2.0's exact output.
- `IK2_GATE_DOWN=1`: only the two down products in one kernel (the mix stays separate), an intermediate step kept as an
  option.

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
