# Benchmarks

## Setup

- AMD Radeon RX 7900 XT 20 GB, connected externally; AMD Radeon 8060S iGPU with 48 GB of shared memory assigned.
  Windows 11, ROCm SDK 7.14.1.
- `Inkling-2-Mini-UD-IQ4_XS.gguf` (and the GSQ-RCO file where stated), every expert in VRAM, FP16 KV cache, greedy decoding, no drafting, one sequence.
- `IK2_GATE_BF16=1`, `IK2_GATE_TILE=64`, context 40,960 unless stated.
- Each run is a fresh process; one unmeasured run per build first warms the file cache; builds alternate between rounds.
  Prompts are fixed token ids, the same for every build (`scripts/bench.py` reproduces this).

## Final build

| Prompt tokens | XT prompt | XT decode | 8060S prompt | 8060S decode |
|---|---|---|---|---|
| 27 | - | 109 | - | 47 |
| 511 | 2,330 | 106 | 1,080 | 46 |
| 4,095 | 3,800 | 102 | 1,620 | 45 |
| 32,760 | 3,490 | 91 | 1,370 | 36 |

Prompt chunk 4,096. Decode is over 200-400 generated tokens.

### GSQ-RCO file (1.4.0)

`Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf`, chunk 4,096, `IK2_GATE_FUSE=16`, 3 alternating rounds, 128 generated tokens (medians,
tok/s). Default (`IK2_REQ8=1` for this file) against `IK2_REQ8=0` (the file's dense weights as they are):

| Prompt | XT prompt | XT decode | XT decode, `IK2_REQ8=0` | 8060S prompt | 8060S decode | 8060S decode, `IK2_REQ8=0` |
|---|---|---|---|---|---|---|
| 27 | - | 103.0 | 77.0 | - | 43.4 | 39.1 |
| 511 | 1,836 | 101.3 | 76.0 | | | |
| 4,095 | 3,206 | 96.2 | 73.1 | 1,524 | 41.4 | 36.3 |
| 32,760 | 2,868 | 84.7 | 65.7 | | | |

Before 1.4.0's token kernels (the same file, the general expert path, no conversion), decode with a short prompt was
65 tok/s on the XT and 34 on the 8060S. Prompt speed is about the same with and without the conversion (XT at 4K:
3,206 against 3,539 in this run, 3,000 against 3,129 in an earlier one; at 32K 2,868 against 2,870).

Reference agreement (llama.cpp b12720, `scripts/reference-gsq.json` and two long prompts), default settings: 186/188
decode and 188/192 prompt positions short, 93/94 and 95/96 long on the XT, 187/188, 189/192, 93/94 and 95/96 on the
8060S (the release binary), no hard miss; with `IK2_REQ8=0`: 186/188, 191/192, 94/94
and 96/96. The IQ3_XXS / IQ4_NL token kernels give the same tokens as the general path (both files).

Quality of the two files against each other (the engine's prompt path, RX 7900 XT, 10 chunks of 2,048 tokens of mixed
code and prose, the second half of each scored as llama-perplexity does): perplexity 1.898 for UD-IQ4_XS and 1.919 for
GSQ-RCO (+1.1%, bootstrap 95% interval +0.7% to +1.6%); the two files pick the same top token at 96.9% of the 10,230
positions. On a small set of maths problems with exact answers and Python functions checked by unit tests (llama.cpp,
greedy), they scored 39/40 and 25/25 (UD) against 39/40 and 24/25 (GSQ).

UD-IQ4_XS file, 1.4.0 against 1.3.0 in the same alternating runs: the same generated tokens at every length on both GPUs,
and the same speed (XT decode 107.4 / 109.3 at 27 tokens, 104.9 / 105.7 at 4K, 89.1 / 89.7 at 32K; prompt 3,619 / 3,626 at
4K; 8060S decode 46.2 / 47.6 and 44.3 / 44.6, prompt 1,679 / 1,654 at 4K).

### Loop gates in one pass (1.3.0)

`IK2_GATE_FUSE=16` against 1.2.0, chunk 4,096, 3 alternating rounds, 128 generated tokens, prompt tok/s (medians):

| Prompt | XT 1.2.0 | XT 1.3.0 | 8060S 1.2.0 | 8060S 1.3.0 |
|---|---|---|---|---|
| 511 | 2,142 | 2,066 | 1,070 | 1,080 |
| 4,095 | 3,444 | 3,556 | 1,564 | 1,672 |
| 32,760 | 3,287 | 3,348 | 1,268 | 1,295 |

Decode after the prompt is the same within noise (XT 103.6 / 101.6 at 4K, 89.7 / 89.5 at 32K; 8060S 43.8 / 43.8 at 4K, 34.6 / 34.9 at 32K). At
511 tokens the XT's spread is larger than the difference (1,860-2,078 for 1.3.0). The generated tokens first differ at
positions 16-65, as expected from another summing order. Reference agreement with `IK2_GATE_FUSE=16`, both GPUs: 188/188
and 192/192 short, 93/94 and 95/96 long, no hard miss (the same as 1.2.0).

### Decode before and after the latency work

| Prompt | XT before | XT after | 8060S before | 8060S after |
|---|---|---|---|---|
| 27 (400 tokens, 3 rounds) | 94.6 94.1 93.8 | 107.4 109.7 109.1 | 42.9 43.6 43.4 | 46.3 47.7 47.4 |
| 511 (400 tokens, 3 rounds) | 91.2 90.6 90.9 | 106.4 107.1 106.3 | 42.2 42.0 41.1 | 45.8 46.0 45.4 |
| 4,095 (200 tokens, 2 rounds) | 86.4 87.5 | 101.6 101.8 | 41.3 39.7 | 45.1 44.9 |
| 32,760 (200 tokens, 2 rounds) | 75.2 76.3 | 90.9 90.2 | 33.7 33.6 | 36.2 36.3 |

Same tokens in every run.

### Warm-up at load

| Prompt | XT off | XT on | 8060S off | 8060S on |
|---|---|---|---|---|
| 128 | 470 | 866 | 389 | 485 |
| 511 | 1,557 | 2,248 | 910 | 1,058 |
| 4,095 | 3,214 | 3,410 | 1,313 | 1,329 |

Same tokens. The warm-up speeds up only the first prompt of a process; later 511-token prompts in a server read
1,890-2,060 tok/s on the XT with or without it.

### Prompt chunk

| Prompt | XT 2,048 | XT 4,096 | 8060S 2,048 | 8060S 4,096 |
|---|---|---|---|---|
| 511 | 2,331 | 2,334 | - | - |
| 4,095 | 3,539 | 3,802 | 1,420 | 1,485 |
| 32,760 | 3,279 | 3,493 | 1,214 | 1,272 |

XT at a 64K context. On the 8060S the generated tokens are the same; on the XT they first differ at token 159 (4,095) and
16 (32,760). In a server (16K-token prompts, median of three after the first), XT: 64K context, chunk 2,048: 3,460;
3,072: 3,600; 4,096: 2,480 (VRAM full). 48K context, chunk 4,096: 3,670. 8060S at 64K: 1,318 and 1,373.

Run-to-run spread on the XT reached 20% for 128-511-token prompts; differences of a few percent there are noise.

### Relative bias in the attention kernel (8060S)

Chunk 4,096, 3 rounds unless stated. The generated tokens are identical with and without it, on both GPUs.

| Prompt | Bias table | Bias in the kernel |
|---|---|---|
| 511 (2 rounds) | 1,021 | 1,082 |
| 4,095 | 1,441 1,426 1,405 | 1,621 1,620 1,628 |
| 32,760 | 1,278 1,236 1,225 | 1,385 1,368 1,317 |
| 4,095, FP8 cache (2 rounds) | 1,451 | 1,663 |

On the RX 7900 XT the same change read 4,095 tokens at 3,387 tok/s against 3,615 with the table, so the XT keeps the table.

### FP8 cache

Decode with `IK2_KV=fp8` (chunk 2,048, context 131,072 on the XT), before and after the matrix-core FP8 attention, 2 rounds:

| Prompt | XT before | XT after | 8060S before | 8060S after |
|---|---|---|---|---|
| 4,095 | 94.0 | 103.2 | 42.4 | 44.2 |
| 32,760 | 89.5 | 95.5 | 37.6 | 38.7 |

In a server on the XT with a 131,072-token context: decode after a 12,628-token prompt 99.9 tok/s (before 89.6), after a
95,029-token prompt 81.5 tok/s (before 75.0). The FP16 cache's tokens and speed are unchanged.

## Other engines

Same GGUF and FP16 KV cache, each engine's own benchmark command, so differences of a few percent are not meaningful.

| | XT prompt 512 | XT prompt 4,096 | XT decode after 4,096 | 8060S prompt 512 | 8060S decode |
|---|---|---|---|---|---|
| strata-inkling 0.1.67 | 2,048-2,064 | 2,028-2,031 | 70.7-70.8 | ~894 | 28.7-31 |
| this build | 2,330 (1,890-2,060 in a server) | 3,800 | ~102 | 1,080 | 45-47 |

The strata-inkling 8060S figures are from an earlier run.

## Quality check

`scripts/gate.py` feeds llama.cpp's greedy tokens for the same file through the engine (teacher forcing) and compares the
engine's top-1 token at every position. Final build, both GPUs:

| Reference | Single-token path | Prompt path |
|---|---|---|
| Short: 4 prompts (`scripts/reference.json`) | 188/188 | 192/192 |
| Long: 991- and 3,854-token prompts | 93/94 | 95/96 |

Prompt chunk 4,096 gives the same scores. On 41 further prompts of 1-3K tokens (3,936 positions, not included), the XT's
prompt path matched 3,838 with chunk 4,096 and 3,840 with chunk 2,048. With `IK2_FLASH2=4` the 8060S matched 3,937 of
4,032 positions against 3,935 for the default kernel. No hard miss anywhere: every miss is a position where llama.cpp's
own top two tokens are within 0.5 logit.

With the FP8 cache (XT): short reference 186/188 single-token and 190/192 prompt, long reference 92/94 and 94/96; on 41
further prompts the prompt path matched 3,811 of 3,936 with one hard miss. The matrix-core FP8 attention against the earlier
FP8 kernel, single-token path: 187 against 186 of 188, 94 against 92 of 94, and 828 against 829 of 855 on nine further prompts.

This check compares top-1 tokens at a few thousand positions. It is not a perplexity or task benchmark.
