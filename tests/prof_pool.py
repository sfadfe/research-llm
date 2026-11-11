"""Breakdown of the pooled-gate overhead at T tokens (default 32768 = one micro-batch of 16x2048).
Run: HIP_VISIBLE_DEVICES=0 .venv/bin/python tests/prof_pool.py"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Train"))
import model as M  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
dev = torch.device("cuda", 0)
torch.manual_seed(0)


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e3


cu = torch.unique(torch.cat([torch.arange(0, T + 1, 2048), torch.randint(1, T, (12,))])).to(dev)
H, D, R = 6, 256, 16
k = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16, requires_grad=True)
w = (torch.randn(H, D, device=dev) * 0.02).requires_grad_()
gr = torch.zeros(H, device=dev, requires_grad=True)
nw = torch.ones(R, device=dev, requires_grad=True)
Ap = (torch.randn(H * R, D, device=dev) / 16).requires_grad_()
U = (torch.randn(12, H * R, device=dev) * 0.02).requires_grad_()
seg = M.padded_segments(1, T, cu, 64, dev)
z32 = torch.cat([k, v], -1).float().detach().requires_grad_()
s32 = (torch.rand(1, T, H, device=dev) * 20 - 10).requires_grad_()
g32 = torch.full((H,), 0.05, device=dev, requires_grad=True)
x32 = torch.randn(1, T, H, R, device=dev, requires_grad=True)
c32 = torch.randn(1, T, H, R, device=dev, requires_grad=True)
cn32 = torch.randn(1, T, H * R, device=dev, requires_grad=True)


def fb(f):
    def run():
        out = f()
        out = out[0] if isinstance(out, tuple) else out
        out.float().sum().backward()
    def fwd():
        with torch.no_grad():
            f()
    return bench(fwd), bench(run)


rows = []
for tag, comp in (("eager", False), ("compiled", True)):
    wrap = (lambda fn: torch.compile(fn, dynamic=False)) if comp else (lambda fn: fn)
    parts = {
        "cat+cast z": wrap(lambda k, v: torch.cat([k, v], -1).float()),
        "score": wrap(lambda z, w: 10 * torch.tanh(torch.einsum("bthd,hd->bth", z, w) / 10)),
        "proj x=A z": wrap(lambda z, A: torch.einsum("bthd,hrd->bthr", z, A.view(H, R, D))),
        "pool_chunks": wrap(lambda z, s, g: M.pool_chunks(z, s, g, seg, 64)),
        "rms_norm": wrap(lambda c, nw: F.rms_norm(c, (R,), nw, 1e-6).flatten(-2)),
        "U (bf16 autocast)": lambda cn, U: F.linear(cn.bfloat16(), U.bfloat16()),
        "full pool_gate": M.pool_gate_compiled if comp else M.pool_gate,
    }
    args = {
        "cat+cast z": (k, v), "score": (z32, w), "proj x=A z": (z32, Ap), "pool_chunks": (x32, s32, g32),
        "rms_norm": (c32, nw), "U (bf16 autocast)": (cn32, U),
        "full pool_gate": (k, v, seg, Ap, w, gr, nw, 64, 10.0, 1e-6),
    }
    for name, fn in parts.items():
        f_ms, fb_ms = fb(lambda fn=fn, a=args[name]: fn(*a))
        rows.append((tag, name, f_ms, fb_ms - f_ms, fb_ms))
for r in rows:
    print(f"{r[0]:9s} {r[1]:18s} fwd {r[2]:6.2f} ms  bwd {r[3]:6.2f} ms  total {r[4]:6.2f} ms", flush=True)

# whole layer A vs B (compiled pooling), forward / backward split
x = torch.randn(1, T, 768, device=dev, requires_grad=True)
for mode, ckpt in (("linear", False), ("pool", False), ("pool", True), ("pool_g0", False), ("pool_s0", False)):
    m = M.GatedDeltaNet(M.ModelConfig(gate_mode=mode, pool_ckpt=ckpt)).to(dev)
    if mode != "linear":
        m.compiled = True
        with torch.no_grad():
            m.ab_c.normal_(0, 0.02)
    def f():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(x, cu, cu.cpu())
    f_ms, fb_ms = fb(f)
    print(f"layer {mode:8s} ckpt={ckpt!s:5s} fwd {f_ms:6.2f} ms  bwd {fb_ms - f_ms:6.2f} ms  total {fb_ms:6.2f} ms", flush=True)

if "--trace" in sys.argv:
    fn = M.pool_gate_compiled
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            c, s = fn(k, v, seg, Ap, w, gr, nw, 64, 10.0, 1e-6)
            c.sum().backward()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
