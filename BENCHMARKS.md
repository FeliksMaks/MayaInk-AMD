# Benchmarks

## Setup

- AMD Radeon RX 7900 XT 20 GB, connected externally; AMD Radeon 8060S iGPU with 48 GB of shared memory assigned.
  Windows 11, ROCm SDK 7.14.1.
- `Inkling-2-Mini-UD-IQ4_XS.gguf`, every expert in VRAM, FP16 KV cache, greedy decoding, no drafting, one sequence.
- `IK2_GATE_BF16=1`, `IK2_GATE_TILE=64`, context 40,960 unless stated.
- Each run is a fresh process; one unmeasured run per build first warms the file cache; builds alternate between rounds.
  Prompts are fixed token ids, the same for every build (`scripts/bench.py` reproduces this).

## Final build

| Prompt tokens | XT prompt | XT decode | 8060S prompt | 8060S decode |
|---|---|---|---|---|
| 27 | - | 109 | - | 47 |
| 511 | 2,330 | 106 | 1,110 | 46 |
| 4,095 | 3,800 | 102 | 1,485 | 45 |
| 32,760 | 3,490 | 91 | 1,270 | 36 |

Prompt chunk 4,096. Decode is over 200-400 generated tokens. With `IK2_FLASH2=4` the 8060S reads 4,095 tokens at 1,701-1,716
and 32,760 at 1,314-1,450 (two rounds).

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

## Other engines

Same GGUF and FP16 KV cache, each engine's own benchmark command, so differences of a few percent are not meaningful.

| | XT prompt 512 | XT prompt 4,096 | XT decode after 4,096 | 8060S prompt 512 | 8060S decode |
|---|---|---|---|---|---|
| strata-inkling 0.1.67 | 2,048-2,064 | 2,028-2,031 | 70.7-70.8 | ~894 | 28.7-31 |
| this build | 2,330 (1,890-2,060 in a server) | 3,800 | ~102 | 1,110 | 45-47 |

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

This check compares top-1 tokens at a few thousand positions. It is not a perplexity or task benchmark.
