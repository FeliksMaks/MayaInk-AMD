# MayaInk-AMD: Inkling 2 Mini on RDNA 3 / 3.5

A HIP engine for Thinking Machines Lab's [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on AMD
RDNA 3 and RDNA 3.5 GPUs under Windows, using Unsloth's `Inkling-2-Mini-UD-IQ4_XS.gguf`. It is a port of the `maya-ik2`
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

For comparison, the same file with strata-inkling 0.1.67 on the RX 7900 XT: 2,030 tok/s prompt at 4,096 tokens and
71 tok/s decode after it. The memory-bandwidth limit for decode is about 200 tok/s on the RX 7900 XT and 62 tok/s on the
8060S (3.88 GB read per token).

The engine's top-1 tokens match llama.cpp's greedy reference on 188 of 188 single-token steps and 192 of 192 prompt
positions (short prompts), and on 93 of 94 and 95 of 96 positions of two long prompts. Every miss is a near tie.

- [OPTIMIZATION.md](OPTIMIZATION.md): what made it faster and what did not.
- [BENCHMARKS.md](BENCHMARKS.md): method, raw numbers, the quality check and its limits.

## Requirements

- Windows 11 and a `gfx1100` (RX 7900 series) or `gfx1151` (Radeon 8060S) GPU. The build also includes `gfx1101`.
- VRAM: 13.1 GiB of experts and 1.0 GiB of dense weights, plus the KV cache (32 KB a token in FP16, 16 KB in FP8).
  On a 20 GB card: up to 64K tokens with the FP16 cache, up to 128K with `-Fp8Kv`.
- Visual Studio 2022 Build Tools with the C++ workload. Visual Studio 18 does not work: ROCm 7.14.1's compiler rejects
  its headers.
- Python 3.10 or newer, and AMD's ROCm SDK 7.14.1 Python packages.
- `Inkling-2-Mini-UD-IQ4_XS.gguf` from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
  (revision `ae9327c5fbc7333116ba32eee78bb5d2bbefae4f`).

## Build

```powershell
python -m venv .venv
.venv\Scripts\pip install cmake ninja numpy
.venv\Scripts\pip install --index-url https://repo.amd.com/rocm/whl-multi-arch "rocm[libraries,devel,device-gfx1100]==7.14.1"
.\build.ps1 -Arch gfx1100
```

For the 8060S install `device-gfx1151` and use `-Arch gfx1151`. With both device packages, `-Arch "gfx1100;gfx1151"`
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
files involved. The model is by Thinking Machines Lab (Apache-2.0) and the GGUF by Unsloth; neither is included.
