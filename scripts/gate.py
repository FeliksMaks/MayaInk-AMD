"""Correctness gate for maya-ik2 against llama.cpp's greedy reference (reference.json).

  python gate.py --exe build\\maya-ik2.exe --model X.gguf --ref reference.json [--mode decode|prefill|both] [--env K=V ...]

decode : the prompt, then the reference's greedy tokens forced one at a time through the real decode path
         (IK2_FORCE + IK2_LOGITS_OUT); step j's logits predict gen[j+1].
prefill: prompt + gen in one prompt pass (--dump), every position's log-probabilities; position P-1+j predicts gen[j].
Per step: engine top-1 == llama's greedy token?  A miss is a near tie when llama's own gap between its top-2 is small
(< 0.5) or the engine's gap between its pick and llama's token is < 0.5.
"""
import argparse, json, os, subprocess, sys, tempfile, time
import numpy as np

V = 200058  # (n_vocab unpadded: the engine writes that many logits)


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def run(cmd, env, log):
    t = time.time()
    cmd = cmd + os.environ.get("GATE_ARGS", "").split()   # (more arguments for the engine, such as --chunk 4096)
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    log.write(" ".join(cmd) + "\n" + p.stdout + "\n")
    if p.returncode != 0:
        print(p.stdout[-3000:])
        sys.exit(f"engine exit {p.returncode}")
    return p.stdout, time.time() - t


def score(lp, ref_gen, ref_top, off):
    """lp [n][V] engine logprobs; row i predicts ref_gen[i + off]."""
    ok = miss_tie = miss_hard = 0
    gaps, dl = [], []
    for i in range(lp.shape[0]):
        j = i + off
        if j >= len(ref_gen):
            break
        want = ref_gen[j]
        got = int(lp[i].argmax())
        top = ref_top[j]
        if top and top[0][0] == want:
            dl.append(abs(lp[i, want] - top[0][1]))
        if got == want:
            ok += 1
            continue
        egap = float(lp[i, got] - lp[i, want])
        lgap = float(top[0][1] - top[1][1]) if len(top) > 1 else 9.0
        gaps.append((j, got, want, round(egap, 3), round(lgap, 3)))
        if egap < 0.5 or lgap < 0.5:
            miss_tie += 1
        else:
            miss_hard += 1
    return ok, miss_tie, miss_hard, gaps, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--mode", default="both")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--log", default="gate.log")
    a = ap.parse_args()
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    ref = json.load(open(a.ref, encoding="utf-8"))
    # (a prompt dump can be several GB: kept beside this script and removed at the end of the run)
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(base, exist_ok=True)
    holder = tempfile.TemporaryDirectory(prefix="ik2gate", dir=base)
    tmp = holder.name
    log = open(a.log, "w", encoding="utf-8")
    tot = {"decode": [0, 0, 0, 0], "prefill": [0, 0, 0, 0]}
    dls = {"decode": [], "prefill": []}
    for n, e in enumerate(ref):
        P, gen, top = e["prompt_ids"], e["gen"], e["top"]
        pf = os.path.join(tmp, f"p{n}.txt")
        if a.mode in ("decode", "both"):
            open(pf, "w").write(",".join(map(str, P)))
            ff = os.path.join(tmp, f"f{n}.txt")
            open(ff, "w").write(",".join(map(str, gen)) + ",")
            lo = os.path.join(tmp, f"l{n}.bin")
            env2 = dict(env, IK2_FORCE=ff, IK2_LOGITS_OUT=lo)
            out, dt = run([a.exe, "--model", a.model, "--ctx", str(a.ctx), "--tokens", "@" + pf, "--gen",
                           str(len(gen))], env2, log)
            lp = logsoftmax(np.fromfile(lo, np.float32).reshape(-1, V))
            r = score(lp, gen, top, 1)
            for k in range(3):
                tot["decode"][k] += r[k]
            tot["decode"][3] += 1
            dls["decode"] += r[4]
            print(f"decode  #{n}: {r[0]}/{r[0] + r[1] + r[2]} top-1, near-tie misses {r[1]}, hard {r[2]} {r[3]}  ({dt:.0f} s)")
        if a.mode in ("prefill", "both"):
            open(pf, "w").write(",".join(map(str, P + gen)))
            dd = os.path.join(tmp, f"d{n}")
            os.makedirs(dd, exist_ok=True)
            out, dt = run([a.exe, "--model", a.model, "--ctx", str(a.ctx), "--tokens", "@" + pf, "--gen", "0",
                           "--dump", dd], env, log)
            lpf = os.path.join(dd, "logprobs.f32")
            want_b = (len(P) + len(gen)) * V * 4
            if os.path.getsize(lpf) < want_b:
                sys.exit(f"GATE INVALID: {lpf} is {os.path.getsize(lpf)} bytes, {want_b} expected (disk full?)")
            lp = np.fromfile(lpf, np.float32).reshape(-1, V)[len(P) - 1:]
            r = score(lp, gen, top, 0)
            del lp
            os.remove(lpf)   # (several GB a prompt: not kept to the end of the run)
            for k in range(3):
                tot["prefill"][k] += r[k]
            tot["prefill"][3] += 1
            dls["prefill"] += r[4]
            print(f"prefill #{n}: {r[0]}/{r[0] + r[1] + r[2]} top-1, near-tie misses {r[1]}, hard {r[2]} {r[3]}  ({dt:.0f} s)")
    for m, (ok, t, h, c) in tot.items():
        if c:
            d = np.array(dls[m])
            print(f"{m}: {ok}/{ok + t + h} = {100 * ok / (ok + t + h):.1f}% top-1, near ties {t}, hard misses {h}; "
                  f"|logp(llama top) - llama| median {np.median(d):.3f} p90 {np.percentile(d, 90):.3f}")


if __name__ == "__main__":
    main()
