"""Torch-profiler breakdown of one training micro-step (fwd+bwd, bf16 autocast, compile_mlps as in train.py)
for a config's [model] section; synthetic tokens, micro_batch x seq_len flattened with doc boundaries (cu).
Run: HIP_VISIBLE_DEVICES=0 .venv/bin/python tests/prof_step.py <config.toml> [<config2.toml> ...]
Prints per-config kernel totals (ms per micro-step), aten-op totals with shapes, and a by-kernel diff vs the first."""
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Train"))
import model as M  # noqa: E402

dev = torch.device("cuda", 0)
NPROF = 3


def batch(mb, L, seed=0):
    gen = torch.Generator().manual_seed(seed)
    T = mb * L
    x = torch.randint(1, 32768, (1, T), generator=gen)
    y = torch.randint(0, 32768, (1, T), generator=gen)
    # doc boundaries: row starts + ~4 random docs per row (similar to packed real data)
    cu = torch.unique(torch.cat([torch.arange(0, T + 1, L), torch.randint(1, T, (4 * mb,), generator=gen)]))
    return x.to(dev), y.to(dev), cu.to(dev).int(), cu.int()


def profile(path):
    cfg = tomllib.load(open(path, "rb"))
    torch.manual_seed(cfg["data"]["seed"])
    m = M.LM(M.ModelConfig(**cfg["model"])).to(dev)
    if cfg["train"]["compile"]:
        m.compile_mlps()
    x, y, cu, cu_cpu = batch(cfg["train"]["micro_batch"], cfg["data"]["seq_len"])

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, y, cu, cu_cpu)
        loss.backward()

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    wall = (time.time() - t) / 5 * 1e3
    m.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 2**30
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        for _ in range(NPROF):
            step()
        torch.cuda.synchronize()
    kern = defaultdict(float)
    cnt = defaultdict(int)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kern[e.name] += e.device_time_total / 1e3 / NPROF
            cnt[e.name] += 1
    ops = []
    for ka in prof.key_averages(group_by_input_shape=True):
        if ka.key.startswith("aten::") and ka.device_time_total > 0:
            ops.append((ka.device_time_total / 1e3 / NPROF, ka.key, str(ka.input_shapes)[:110], ka.count // NPROF))
    ops.sort(reverse=True)
    return wall, mem, dict(kern), dict(cnt), ops


def short(n, w=95):
    return n if len(n) <= w else n[:w - 3] + "..."


if __name__ == "__main__":
    res = {}
    for p in sys.argv[1:]:
        name = Path(p).stem
        wall, mem, kern, cnt, ops = profile(p)
        res[name] = kern
        tot = sum(kern.values())
        print(f"\n=== {name}: wall {wall:.1f} ms/micro-step (fwd+bwd, 5 avg), kernel sum {tot:.1f} ms, peak {mem:.2f} GiB")
        for n, ms in sorted(kern.items(), key=lambda kv: -kv[1])[:30]:
            print(f"  {ms:8.2f} ms  x{cnt[n] // NPROF:4d}  {short(n)}")
        print(f"  -- aten ops by device time (incl. children), top 30")
        for ms, k, shp, c in ops[:30]:
            print(f"  {ms:8.2f} ms  x{c:4d}  {k:28s} {shp}")
        torch.cuda.empty_cache()
    names = list(res)
    base = res[names[0]]
    for other in names[1:]:
        o = res[other]
        print(f"\n=== diff {other} - {names[0]} (by kernel name; total {sum(o.values()) - sum(base.values()):+.1f} ms)")
        keys = set(base) | set(o)
        diff = sorted(((o.get(k, 0) - base.get(k, 0), k) for k in keys), key=lambda d: -abs(d[0]))
        for d, k in diff[:35]:
            print(f"  {d:+8.2f} ms  (A {base.get(k, 0):7.2f} / B {o.get(k, 0):7.2f})  {short(k)}")
