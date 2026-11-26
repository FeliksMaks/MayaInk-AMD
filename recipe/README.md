# Recipes for local-ai-registry

Draft entries for [sybil-solutions/local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) in its
recipe format. The registry accepts only recipes written by its own tool,
`lab.py try`, after a run passes its six checks (load, chat, reasoning, tools, context, and at least 15 tok/s decode).
Their `proof` arrays are filled by that tool.

| File | Card | Context | KV cache | Chunk |
|---|---|---|---|---|
| `rx-7900-xt-20gb/inkling-2-mini.maya-ik2-rdna.48k.json` | RX 7900 XT 20 GB | 49,152 | FP16 | 4,096 |
| `rx-7900-xt-20gb/inkling-2-mini.maya-ik2-rdna.128k.json` | RX 7900 XT 20 GB | 131,072 | FP8 | 2,048 |
| `radeon-8060s-64gb/inkling-2-mini.maya-ik2-rdna.128k.json` | Radeon 8060S | 131,072 | FP16 | 4,096 |

`engines/maya-ik2-rdna.json` describes the launch in the shape of the registry's engine profiles.

The card ids are the registry's. The 8060S was measured with 48 GB of shared memory assigned to the GPU; the registry's
card is the 64 GB variant.

## Submitting a recipe

1. The recipe files pin the engine to release `v1.1.0` of this repository.
2. On the machine with the card, start the server (`serve.ps1`, see the main README) with the recipe's settings.
3. In a clone of the registry, run its tool against that server:
   `python3 lab/lab.py try --on endpoint --endpoint http://127.0.0.1:18000/v1 ...` (see the registry's README for the
   model and card arguments). It writes the recipe with its proof only if all six checks pass.
4. Follow the registry's contribution guide: `make`, `make check`, and one pull request with the evidence the tool
   printed. The registry's [recipe kit](https://github.com/sybil-solutions/local-ai-recipe-kit) describes its
   measurement protocol.
