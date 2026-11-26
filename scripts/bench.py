"""Prompt and decode speed at several prompt lengths; several builds alternate and their generated ids are compared.

  python bench.py --model X.gguf --gpu 7900 --exe build\\maya-ik2.exe [--exe other.exe ...] [--lengths 27,511,4095] [--rounds 3] [--chunk 4096]
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gpu", default="7900")
ap.add_argument("--exe", action="append", required=True)
ap.add_argument("--lengths", default="27,511,4095")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--gen", type=int, default=200)
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--chunk", type=int, default=4096)
a = ap.parse_args()
env = dict(os.environ, IK2_UNSUPPORTED="1")
env.setdefault("IK2_GATE_BF16", "1")
env.setdefault("IK2_GATE_TILE", "64")
env.pop("HIP_VISIBLE_DEVICES", None)
env["PATH"] = os.path.join(sys.prefix, "Lib", "site-packages", "_rocm_sdk_devel", "bin") + os.pathsep + env["PATH"]
idx = subprocess.run([sys.executable, str(here / "hip_gpu_index.py"), a.gpu], env=env, capture_output=True, text=True)
if idx.returncode:
    sys.exit(f"no GPU whose name contains {a.gpu}")
env["HIP_VISIBLE_DEVICES"] = idx.stdout.strip()
base = [t for e in json.loads((here / "reference.json").read_text(encoding="utf-8")) for t in e["prompt_ids"] + e["gen"]]
first = [int(t) for t in (here / "prompt27.txt").read_text().split(",")]


def run(exe, f, gen):
    o = subprocess.run([exe, "--model", a.model, "--ctx", str(a.ctx), "--chunk", str(a.chunk), "--tokens", "@" + f, "--gen", str(gen)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace").stdout
    p = re.search(r"^prompt: .*?\(([\d.]+) tok/s\)", o, re.M)
    d = re.search(r"^generation: .*?\(([\d.]+) tok/s\)", o, re.M)
    t = re.search(r"^generated: (.*)$", o, re.M)
    if not (p and d and t):
        sys.exit(f"{exe} failed:\n{o[-800:]}")
    return float(p[1]), float(d[1]), t[1].split()


with tempfile.TemporaryDirectory() as tmp:
    for n in (int(x) for x in a.lengths.split(",")):
        f = os.path.join(tmp, f"p{n}.txt")
        ids = first if n == 27 else (base * (n // len(base) + 1))[:n]
        Path(f).write_text(",".join(map(str, ids)))
        got = {e: [] for e in a.exe}
        for e in a.exe:
            run(e, f, 8)
        for r in range(a.rounds):
            for e in a.exe if r % 2 == 0 else a.exe[::-1]:
                got[e].append(run(e, f, a.gen))
        ref = got[a.exe[0]][0][2]
        for e in a.exe:
            v = got[e]
            print(f"{n:6d} {e}: prompt {statistics.median(x[0] for x in v):.0f}  decode {statistics.median(x[1] for x in v):.1f}"
                  f"  ids {'same' if all(x[2] == ref for x in v) else 'DIFFER'}", flush=True)
