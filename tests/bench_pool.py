"""Benchmark one GDN layer fwd+bwd per gate mode / pooling variant. Run: .venv/bin/python tests/bench_pool.py [T]"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Train"))
import model as M  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
dev = torch.device("cuda", 0)


def bench(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e3, torch.cuda.max_memory_allocated() / 2**30


cu = torch.unique(torch.cat([torch.arange(0, T + 1, 2048), torch.randint(1, T, (12,))])).to(dev)
x = torch.randn(1, T, 768, device=dev, requires_grad=True)


def layer_step(m):
    def f():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(x, cu, cu.cpu())
        y.float().sum().backward()
    return f


for mode in ("linear", "pool", "pool_g0", "pool_s0"):
    for compiled in ((False,) if mode == "linear" else (False, True)):
        for ckpt in ((False,) if mode == "linear" else (True, False)):
            m = M.GatedDeltaNet(M.ModelConfig(gate_mode=mode, pool_ckpt=ckpt)).to(dev)
            if mode != "linear":
                m.compiled = compiled
                with torch.no_grad():
                    m.ab_c.normal_(0, 0.02)
            ms, mem = bench(layer_step(m))
            print(f"layer {mode:8s} compiled={compiled!s:5s} ckpt={ckpt!s:5s} {ms:7.2f} ms  peak {mem:.2f} GB", flush=True)
            del m
            torch.cuda.empty_cache()

# isolated pooling core
k = torch.randn(1, T, 6, 128, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, T, 6, 128, device=dev, dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(6, 256, device=dev, requires_grad=True)
gr = torch.zeros(6, device=dev, requires_grad=True)
nw = torch.ones(256, device=dev, requires_grad=True)
seg = M.padded_segments(1, T, cu, 64, dev)
for name, fn in (("eager", M.pool_gate), ("compiled", M.pool_gate_compiled)):
    def f(fn=fn):
        c, s = fn(k, v, seg, w, gr, nw, 64, 10.0, 1e-6)
        c.sum().backward()
    def g(fn=fn):
        with torch.no_grad():
            fn(k, v, seg, w, gr, nw, 64, 10.0, 1e-6)
    print(f"pool_gate {name:8s} fwd+bwd {bench(f)[0]:7.2f} ms  fwd {bench(g)[0]:7.2f} ms", flush=True)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        c, s = M.pool_gate(k, v, seg, w, gr, nw, 64, 10.0, 1e-6)
        c.sum().backward()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
