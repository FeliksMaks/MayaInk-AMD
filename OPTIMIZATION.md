# Where the speed came from

Numbers are tokens per second, medians of alternating runs of two builds (or of one build with a setting on and off), each
run a fresh process. "Same tokens" means the generated token ids were identical. XT is the RX 7900 XT, 8060S the iGPU.

## The model as a workload

Inkling 2 Mini has 24 layers; layers 6-17 run three times, so a token makes 48 layer visits, each reading 8 of 128
experts. A token reads about 3.88 GB: 1.58 GB of NVFP4 experts (layers 0-19), about 0.15 GB of experts in other formats
(layers 20-23: Q5_K, Q4_K, IQ4_XS, Q5_0, IQ4_NL), 1.27 GB of Q8_0 attention weights, a 0.38 GB Q6_K output head and
0.39 GB of F32 tensors. At 775 GB/s (XT) that is a limit of about 200 tok/s; at 241 GB/s (8060S) about 62 tok/s.

The upstream engine was written for CUDA. Its tensor-core kernels use `mma.sync`, which has no RDNA equivalent; the port
first emulated it with lane shuffles (`ik2_hip.cuh`) and then replaced the hot paths with kernels for RDNA's WMMA
instructions (`ik2_rdna.inc`).

## Decode

| Change | XT | 8060S | Tokens |
|---|---|---|---|
| Small copies (embedding row, positions, states) as kernels instead of runtime copies | not measured alone | not measured alone | same |
| Decode attention on the matrix cores instead of emulated tensor-core code | 91.3 -> 93.2 | 42.6 -> 43.6 | differ late |
| A token's expert gate/up and down in one dispatch | 94.7 -> 95.6 | 44.1 -> 44.2 | same |
| Router product with 24 loads in flight per lane | 93.5 -> 100.4 | measured with the next row | same |
| The same for loop gates, attention sinks, MoE combine and attention merge | 99.7 -> 108.6 | 43.4 -> 47.4 | same |
| FP8 cache: decode attention on the matrix cores, FP8 values turned into FP16 by moving their bits | 94.0 -> 103.2 (4K), 89.5 -> 95.5 (32K) | 42.4 -> 44.2, 37.6 -> 38.7 | differ; gate equal |

The last two rows are the largest gain and the simplest change. A token runs several hundred small kernels, and many
of them were loops of this form:

```c
for (int i = lane; i < K; i += 32) a += h2f(w[i]) * x[i];   // 72 steps, one load in flight per step
```

Each step waits for its load, so a small product costs a chain of memory latencies: the router took about 13 us per
layer visit for 59 KB of weights. Loading a batch of steps first and then adding them in the same order gives the same
result bit for bit and removes most of the wait:

```c
for (int b = 0; b < 24; ++b) wv[b] = w[i + 32 * b], xv[b] = x[i + 32 * b];   // all loads issued
for (int b = 0; b < 24; ++b) a += h2f(wv[b]) * xv[b];                          // same order of additions
```

With an FP8 cache, decode attention had used the emulated tensor-core code. The FP8 values are now made FP16 in registers by
moving their exponent and mantissa bits into place, which gives the exact value divided by 256 (subnormals included); the
factor 256 is folded into the logit and value scales. A first version that converted each value in software was slower than
the emulated kernel (37.4 -> 28.4 tok/s at 32K on the 8060S).

Such kernels are found by comparing each kernel group's time with its bytes. Timing came from launching a group twice
and dividing the difference by the number of visits.

## Prompt

| Change | XT | 8060S |
|---|---|---|
| A layer's experts in one launch on the matrix cores, instead of one expert at a time | 650 -> ~1,300 (4K) | 318 -> ~590 |
| Projections and other dense products on the matrix cores, read directly from their quantized blocks | ~1,300 -> ~2,400 | ~590 -> ~940 |
| Expert tiles of 128 rows, then 256 weight columns; a loop-gate prompt kernel; faster relative bias and copies | not measured alone | not measured alone |
| Dense products in 128-row tiles; attention bias loads issued before the tile's barrier | 2,790 -> 2,930 | 1,053 -> 1,267 |
| Layers 20-23 (five other formats) decoded inside the same tiles | 2,510 -> 3,102 (4K), 1,207 -> 1,494 (511) | 1,305 -> 1,359, 784 -> 998 |
| Warm-up at load: buffers at full size and first kernel launches (first prompt only) | 1,557 -> 2,248 (511) | 910 -> 1,058 |
| Expert inputs read in place; fused projections written to their final place | 3,457 -> 3,490 (4K) | 1,355 -> 1,382 |
| Relative-position bias computed in the attention kernel instead of a table (8060S only; same tokens) | 3,615 -> 3,387: not used | 1,426 -> 1,621 (4K), 1,236 -> 1,368 (32K) |
| Loop gates: both down products and the mix in one kernel, input and message read once (1.3.0) | 3,444 -> 3,556 (4K), 3,287 -> 3,348 (32K) | 1,564 -> 1,672, 1,268 -> 1,295 |
| Prompt chunks of 4,096 tokens instead of 2,048 (a setting) | 3,539 -> 3,802 (4K), 3,279 -> 3,493 (32K) | 1,420 -> 1,485, 1,214 -> 1,272 |

The prompt attention adds a learned bias that depends on the distance between query and key. It was computed as a separate
[tokens][heads][1,024 distances] F32 table, about 400 MB per layer visit at 4K tokens, written by one kernel and read by the
attention. The attention kernel now computes each value itself from the query's 16 relative values (kept in shared memory),
in the table kernel's order of operations, so the results are identical. The 8060S, which waits for memory, gains 11-14%;
the RX 7900 XT, which does not, loses 6%, so it keeps the table.

Layers 20-23 are 4 of 48 visits but took 40% of the expert time at 511 tokens while each of their experts ran as its own
decode and product. The first expert launch of a process took 55 ms on the XT against about 2 ms later, which is what
the warm-up removes from the first prompt.

## Tried and dropped

| Attempt | Result |
|---|---|
| Decode experts: both rows' weights in flight | 103.3 vs 109.1 |
| Decode experts: down blocks fetched while waiting for gate/up | 106.0-107.8 vs 108.8 |
| Q8 projections: 6, 9 or 18 steps in flight | 107.1-108.2 vs 108.4 |
| Head, conv epilogues, norms, gate weights loaded before their barrier | 108.0-109.1 vs 109.3 |
| Flushing the queue in the middle of a token | 91.6-94.5 vs 95.9 |
| Emulated tensor-core expert kernels | 61 vs 75 |
| Prompt expert tiles: weights two steps ahead | 603 vs 684 (128 tokens), 1,675 vs 1,725 (511) |
| Expert tiles of 128 weight columns for experts with few rows | 0.125-0.133 s vs 0.105 s of expert time at 128 tokens |
| Expert tiles of 256 rows x 128 columns (8060S, 4K) | 1,678 vs 1,701 |
| A separate kernel for full 128-row expert tiles (8060S, 4K) | 1,647 vs 1,710 |
| Expert tiles limited to 192 registers (8060S, 4K) | 1,673 vs 1,716 |
| Dense tiles of 256 weight rows (8060S, 4K) | 1,678 vs 1,705 |
| Loop gates as BF16, convs and combine with batched loads in the prompt (8060S, 4K) | 1,410-1,423 vs 1,402, tokens differ |
| Prompt attention with one lane per query (`IK2_FLASH2`), XT | 3,084 -> 3,023 |
| Expert tiles of 64 rows on a 4-warp block for experts with few rows | XT 2,186 -> 1,991 (511), 8060S 1,126 -> 993 |
| Expert tiles of 256 rows on 16 warps | XT 3,664 -> 3,618 (4K), 3,436 -> 3,378 (32K) |
| Expert tiles with two shared-memory buffers (one barrier a step) | XT 3,682 -> 3,512 (4K), 8060S 1,668 -> 1,530 |
| Expert kernel asking for 8 waves per SIMD | XT 3,674 -> 3,675, 8060S 1,675 -> 1,664 |
| Loop gate up kernel over 32 or 64 tokens a block instead of 16 | XT 3,858 -> 3,821 / 3,797 (4K) |

The lane-per-query attention is faster on the 8060S (1,382 -> 1,562 at 4K with a 2,048 chunk). Its tokens differ from the
default kernel's, and on 4,032 reference positions it matched llama.cpp on 3,937 against 3,935 for the default, so it is
offered as an option.
