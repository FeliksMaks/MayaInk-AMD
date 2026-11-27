# MayaInk-AMD: Inkling 2 Mini on RDNA 3 / 3.5

A HIP engine for Thinking Machines Lab's [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on AMD
RDNA 3 and RDNA 3.5 GPUs under Windows, using Unsloth's `Inkling-2-Mini-UD-IQ4_XS.gguf` or ISTA-DASLab's smaller
`Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf`. It is a port of the `maya-ik2`
engine from [alesha-pro/maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini) (commit `c520fd1`, written
for CUDA on an RTX 3090), with new kernels for AMD's matrix cores, support for the extra weight formats of the IQ4_XS
file, and a reworked single-token path.

## Results

One chat, every expert in VRAM, greedy decoding, FP16 KV cache, prompt chunk 4,096. Medians of 2-3 runs, tokens per second.

| | RX 7900 XT 20 GB | Radeon 8060S (iGPU) |
|---|---|---|
| Decode, 27-token prompt | 109 | 47 |
| Decode after 511 tokens | 106 | 46 |
| Decode after 4,095 tokens | 102 | 45 |
| Decode after 32,760 tokens | 91 | 36 |
| Prompt, 511 tokens | 2,330 | 1,080 |
| Prompt, 4,095 tokens | 3,800 | 1,620 |
| Prompt, 32,760 tokens | 3,490 | 1,370 |

Version 1.3.0 against 1.2.0 in the same alternating runs (prompt tok/s): RX 7900 XT 3,444 -> 3,556 at 4K tokens and
3,287 -> 3,348 at 32K; Radeon 8060S 1,564 -> 1,672 at 4K and 1,268 -> 1,295 at 32K. The table above is from earlier
runs; the RX 7900 XT's speed moves by a few percent from one session to the next, which is why versions are compared
within one run.

For comparison, the same file with strata-inkling 0.1.67 on the RX 7900 XT: 2,030 tok/s prompt at 4,096 tokens and
71 tok/s decode after it. The memory-bandwidth limit for decode is about 200 tok/s on the RX 7900 XT and 62 tok/s on the
8060S (3.88 GB read per token).

The engine's top-1 tokens match llama.cpp's greedy reference on 188 of 188 single-token steps and 192 of 192 prompt
positions (short prompts), and on 93 of 94 and 95 of 96 positions of two long prompts. Every miss is a near tie.

## Progress, 1.0 to 1.4

Each release was measured against the one before it in the same alternating runs and checked against the llama.cpp
reference before it was published.

| Version | Change | Measured effect |
|---|---|---|
| 1.0.0 | First release: matrix-core prompt kernels, batched single-token path | RX 7900 XT 109 tok/s decode, 3,800 tok/s prompt at 4K; Radeon 8060S 47 / 1,485 |
| 1.1.0 | FP8 KV cache: decode attention on the matrix cores | With `-Fp8Kv` on the RX 7900 XT: 94 -> 103 tok/s after 4K tokens, 75 -> 81 after 95K |
| 1.2.0 | 8060S: relative bias computed inside the prompt attention | 8060S prompt 1,426 -> 1,621 tok/s at 4K, 1,236 -> 1,368 at 32K |
| 1.3.0 | Loop gates' down products and mix in one pass | Prompt at 4K: RX 7900 XT +3.3%, 8060S +6.9% |
| 1.4.0 | ISTA-DASLab's GSQ-RCO file; its dense weights as Q8_0, compiled IQ3_XXS / IQ4_NL expert kernels | GSQ decode RX 7900 XT 65 -> 103 tok/s, 8060S 34 -> 43; RX 7800 XT 16 GB: 83 tok/s |

For reference, the same UD file on strata-inkling 0.1.67 (RX 7900 XT): 2,030 tok/s prompt at 4K and 71 tok/s decode.

## Two GGUF files

Since 1.4.0 the engine also runs ISTA-DASLab's
[Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf](https://huggingface.co/ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF) (11.4 GB, against
15.7 GB for Unsloth's UD-IQ4_XS). The engine reads the file's layout from its metadata; nothing has to be converted.

GSQ-RCO file with 1.4.0, same settings as above, medians of 3 alternating runs (tok/s):

| | RX 7900 XT 20 GB | Radeon 8060S (iGPU) |
|---|---|---|
| Decode, 27-token prompt | 103 | 43 |
| Decode after 511 tokens | 101 | |
| Decode after 4,095 tokens | 96 | 41 |
| Decode after 32,760 tokens | 85 | |
| Prompt, 511 tokens | 1,840 | |
| Prompt, 4,095 tokens | 3,210 | 1,520 |
| Prompt, 32,760 tokens | 2,870 | |

On a 16 GB RX 7800 XT (`gfx1101`) only the GSQ file fits in VRAM: 82.9 tok/s decode with a short prompt, 77.8 after
4,095 tokens, 2,510 tok/s prompt at 4K. The UD file's 13.1 GiB of experts do not fit beside the rest, so they are read
from system memory through a VRAM cache: about 7 tok/s (one run). The smaller file's gain is that it fits, not that it
reads fewer bytes.

Its top-1 tokens match llama.cpp b12720's greedy reference on 186 of 188 single-token steps and 188 of 192 prompt
positions (short prompts), and 93 of 94 and 95 of 96 on the long ones; every miss is a near tie.

**Why the smaller file is not faster.** Decode here is not limited by how many bytes a token reads. Inkling 2 Mini
runs its middle layers three times, so a token is 48 layer passes one after another, each a chain of small kernels.
At 100 tok/s the RX 7900 XT reads about 1.1 GB a token, an eighth of its memory bandwidth: the time goes to the fixed
cost of each step, and a smaller file shortens only the part that was never the bottleneck. The GSQ file's formats
also cost more to unpack than the UD file's: its experts are 3-bit i-quants (IQ3_XXS, values looked up in a grid
with separate sign bits) where the UD file has NVFP4 (two instructions a pair of values), and its attention weights are
a mix of Q4_K, Q5_K, Q6_K and BF16 where the UD file has Q8_0 throughout.

What 1.4.0 does about it: those attention weights are converted to Q8_0 while the file loads, so they take the same
single kernel as the UD file's (decode 65 -> 82 tok/s on the RX 7900 XT); the expert kernels have compiled versions
for IQ3_XXS and IQ4_NL (82 -> 100 tok/s, identical output). The conversion changes values by less than 0.5% and the
output slightly; `IK2_REQ8=0` keeps the file's weights as they are (about 76 tok/s), with output that matches the
earlier engine exactly. Load takes about 10 s longer with the conversion.

- [OPTIMIZATION.md](OPTIMIZATION.md): what made it faster and what did not.
- [BENCHMARKS.md](BENCHMARKS.md): method, raw numbers, the quality check and its limits.

## Requirements

- Windows 11 and a `gfx1100` (RX 7900 series), `gfx1101` (RX 7800 / 7700 XT, tested on an RX 7800 XT with the GSQ file)
  or `gfx1151` (Radeon 8060S) GPU.
- VRAM: 13.1 GiB of experts and 1.0 GiB of dense weights, plus the KV cache (32 KB a token in FP16, 16 KB in FP8).
  On a 20 GB card: up to 64K tokens with the FP16 cache, up to 128K with `-Fp8Kv`.
- Visual Studio 2022 Build Tools with the C++ workload. Visual Studio 18 does not work: ROCm 7.14.1's compiler rejects
  its headers.
- Python 3.10 or newer, and AMD's ROCm SDK 7.14.1 Python packages.
- `Inkling-2-Mini-UD-IQ4_XS.gguf` from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
  (revision `ae9327c5fbc7333116ba32eee78bb5d2bbefae4f`), or `Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf` from
  [ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF) (revision
  `f0da1097633dd7c2f5733e4077cff861be122245`). The GSQ file needs 8.9 GiB for its experts and 1.2 GiB for its dense
  weights.

## Build

```powershell
python -m venv .venv
.venv\Scripts\pip install cmake ninja numpy
.venv\Scripts\pip install --index-url https://repo.amd.com/rocm/whl-multi-arch "rocm[libraries,devel,device-gfx1100]==7.14.1"
.\build.ps1 -Arch gfx1100
```

For the 8060S install `device-gfx1151` and use `-Arch gfx1151`; for an RX 7800 XT or 7700 XT, `device-gfx1101` and
`-Arch gfx1101` (the run needs that package's rocBLAS kernels too). With both device packages, `-Arch "gfx1100;gfx1151"`
builds one binary for both. The output is `build\maya-ik2.exe`.

## Run from the command line

```powershell
.\run.ps1 -Model C:\models\Inkling-2-Mini-UD-IQ4_XS.gguf -Gpu 7900 -Tokens 200
```

`-Gpu` is part of the GPU's name. HIP's device order can change between boots, so the scripts select a GPU by name.
This form takes token ids and prints token ids and speeds.

## Run as a chat server

The OpenAI and Anthropic compatible server is the Python server of the upstream repository. This repository does not
include it; `serve.ps1` runs it with the engine built here.

```powershell
git clone https://github.com/alesha-pro/maya-inkling-mini C:\src\maya-inkling-mini
git -C C:\src\maya-inkling-mini checkout c520fd1
.venv\Scripts\pip install jinja2 regex pyyaml tqdm requests pillow psutil "jsonschema>=4.18,<5" huggingface_hub
.\serve.ps1 -Upstream C:\src\maya-inkling-mini -Model C:\models\Inkling-2-Mini-UD-IQ4_XS.gguf -Gpu 7900 -Context 49152
```

The first start writes the tokenizer files to `build\tokenizer`. The server listens on `http://127.0.0.1:18000/v1`
(model `inkling-2-mini`) and serves a chat page at `http://127.0.0.1:18000/`. For a 128K context on a 20 GB card:

```powershell
.\serve.ps1 -Upstream C:\src\maya-inkling-mini -Model C:\models\Inkling-2-Mini-UD-IQ4_XS.gguf -Context 131072 -Fp8Kv
```

On an RX 7900 XT a 95,000-token prompt took 39 s with this setting, and decode after it ran at 81 tok/s. The server is
used for text chat, one conversation at a time.

## Settings

| Setting | Default | Effect |
|---|---|---|
| `--chunk N` | 2048 (4096 in the scripts) | Prompt tokens per pass. 4,096 is 4-8% faster than 2,048 and needs about 1.3 GB more VRAM. `serve.ps1` uses 4,096 up to a 48K context, 3,072 up to 64K and 2,048 above, because a full card gets slow: at 64K on 20 GB, 4,096 read 16K prompts at 2,480 tok/s against 3,460 with 2,048. |
| `IK2_KV=fp8` (`-Fp8Kv`) | FP16 | Half the KV memory, and faster decode at long contexts. On the long reference it scored 94 of 96 on the prompt path, against 95 of 96 for FP16. |
| `IK2_GATE_BF16=1`, `IK2_GATE_TILE=64` | set by the scripts | Loop-gate weights read as BF16 (they hold exact BF16 values), 64 channels a block. |
| `IK2_GATE_FUSE=16` (`-NoGateFuse` turns it off) | set by the scripts | The loop gates' down products and mix in one pass, 16 tokens a block (`=1`: 32). Prompts 2-7% faster; another summing order, so some tokens differ from 1.2.0, with the same reference agreement. |
| `IK2_REQ8=0` / `1` | 1 for GSQ-RCO files, 0 otherwise | Dense weights of other block types (the GSQ file's K-quants and BF16) converted to Q8_0 at load: faster decode, slightly different output, about 10 s more load time. |
| `IK2_TOKTY=0` | 1 | The single-token expert kernels for IQ3_XXS and IQ4_NL take the general path instead of their compiled versions. Same output, slower. |
| `IK2_GATE_DOWN=1` | 0 | Only the loop gates' two down products in one kernel. Superseded by `IK2_GATE_FUSE`. |
| `IK2_FLASH_INL1=0` / `1` | 1 on the 8060S, 0 on the RX 7900 XT | Prompt attention computes its relative-position bias itself instead of reading a precomputed table. Same tokens either way; faster on the 8060S, slower on the RX 7900 XT. |
| `IK2_FLASH2=4` | 0 | Prompt attention with one lane per query: faster prompts on the 8060S, no gain on the RX 7900 XT. Its tokens differ from the default; its agreement with the reference is the same (BENCHMARKS.md). |
| `IK2_PRIME=0` | 1 | No warm-up at load (0.4 s). The first prompt then pays for buffer growth and first kernel launches. |
| `IK2_PIPE=0` | 1 | A token's expert gate/up and down in two launches instead of one. |
| `IK2_RDNA_MIX=0` | 1 | The prompt runs layers 20-23 one expert at a time. |
| `IK2_NOGATHER=0`, `IK2_NOUNPACK=0` | 1 | Copy expert inputs together, and fused projections apart, in separate passes. |
| `IK2_ATTN_WMMA=0`, `IK2_ROUTER_FAST=0` | 1 | Earlier decode attention and router kernels. |
| `IK2_ATTN_WMMA8=0` | 1 | With an FP8 cache, the earlier decode attention kernel. |

`IK2_PIPE` assumes that the blocks of one dispatch start in order. Its wait is bounded and writes NaN if that assumption
fails, so a failure is visible rather than a hang.

## Check a build

```powershell
.venv\Scripts\python scripts\gate.py --exe build\maya-ik2.exe --model <gguf> --ref scripts\reference.json
.venv\Scripts\python scripts\bench.py --exe build\maya-ik2.exe --model <gguf> --gpu 7900
```

`gate.py` teacher-forces llama.cpp's greedy tokens through both paths of the engine and reports top-1 agreement.
For the GSQ-RCO file use `--ref scripts\reference-gsq.json` (the same prompts, from llama.cpp b12720).
`bench.py` alternates builds and reports prompt and decode speed and whether their generated tokens are identical.

## Repository layout

| Path | Contents |
|---|---|
| `src/ik2/` | The engine. `ik2_rdna.inc` (matrix-core kernels) and `ik2_hip.cuh` are AMD-specific; the other files are the upstream engine with this port's changes, marked `(AMD port)` in the source. |
| `include/strata/` | Headers from Project Maya and Strata: GGUF reader, HIP compatibility layer, RDNA WMMA helpers. |
| `amd/compat/` | Two small headers for CUDA runtime names that HIP does not provide. |
| `scripts/` | Correctness gate, benchmark, GPU lookup, reference tokens. |
| `recipe/` | A draft entry for [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry). |

## How it was made

The port, the kernels and the measurements were done with the help of AI coding assistants: in part with a Grok model,
and in part with Claude Opus 5.5. Every change was built, benchmarked and checked against the llama.cpp reference on the
two GPUs above before it was kept.

## Credits and licences

Code: MIT (see [LICENSE](LICENSE)), as the upstream engine. Parts of `src/ik2/ik2_rdna.inc` follow designs from CIRU AI's
Apache-2.0 project; those kernels are marked in the source. [NOTICE](NOTICE) lists every source with its version and the
files involved. `third_party/ggml/ggml-common.h` (the i-quant grids) is ggml's, MIT. The model is by Thinking Machines
Lab (Apache-2.0) and the GGUF files by Unsloth and ISTA-DASLab; none is included.
