"""One GatedDeltaNet layer at T tokens (default 32768), bf16 autocast: fwd / bwd / total ms and peak memory,
for linear, pool torch-compiled (ckpt on/off) and pool triton (bt=32, 64). Same harness as tests/prof_pool.py.
Run: HIP_VISIBLE_DEVICES=0 .venv/bin/python tests/bench_layer_impl.py [T]"""
import functools
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Train"))
import model as M  # noqa: E402
import pool_triton as PT  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
dev = torch.device("cuda", 0)
torch.manual_seed(0)
ORIG = M.pool_gate_triton


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e3


def peak(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**30


cu = torch.unique(torch.cat([torch.arange(0, T + 1, 2048), torch.randint(1, T, (12,))])).to(dev)
cu_cpu = cu.cpu()
x = torch.randn(1, T, 768, device=dev, requires_grad=True)

rows = []
for tag, mode, impl, ckpt, bt in (
    ("linear", "linear", None, False, None),
    ("pool torch compiled ckpt=off", "pool", "torch", False, None),
    ("pool torch compiled ckpt=on", "pool", "torch", True, None),
    ("pool triton bt=32", "pool", "triton", False, 32),
    ("pool triton bt=64", "pool", "triton", False, 64),
    ("pool_g0 triton bt=32", "pool_g0", "triton", False, 32),
    ("pool_s0 triton bt=32", "pool_s0", "triton", False, 32),
    ("pool_s0 triton bt=64", "pool_s0", "triton", False, 64),
):
    M.pool_gate_triton = ORIG if bt is None else functools.partial(PT.pool_gate_triton, bt=bt)
    cfg = M.ModelConfig(gate_mode=mode, pool_ckpt=ckpt, pool_impl=impl or "torch")
    m = M.GatedDeltaNet(cfg).to(dev)
    if mode != "linear":
        m.compiled = impl == "torch"
        with torch.no_grad():
            m.ab_c.normal_(0, 0.02)

    def f():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(x, cu, cu_cpu)

    def fwd():
        with torch.no_grad():
            f()

    def fb():
        f().float().sum().backward()

    try:
        f_ms, fb_ms = bench(fwd), bench(fb)
        x.grad = None
        m.zero_grad(set_to_none=True)
        mem = peak(fb)
    except Exception as e:
        print(f"layer {tag:30s} ERROR {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=3)
        continue
    print(f"layer {tag:30s} fwd {f_ms:6.2f} ms  bwd {fb_ms - f_ms:6.2f} ms  total {fb_ms:6.2f} ms  peak {mem:5.2f} GiB",
          flush=True)
    del m
    x.grad = None
    torch.cuda.empty_cache()
M.pool_gate_triton = ORIG

# triton pooled gate alone (no layer), bf16 kv slice of qkv like in the layer
H, R = 6, 16
qkv = torch.randn(1, T, 3 * H * 128, device=dev, dtype=torch.bfloat16)
seg = M.segment_ids(1, T, cu, dev)
A = (torch.randn(H * R, 256, device=dev) / 16).requires_grad_()
w = (torch.randn(H, 256, device=dev) * 0.02).requires_grad_()
g = torch.full((H,), 0.05, device=dev, requires_grad=True)
nw = torch.ones(R, device=dev, requires_grad=True)
for bt in (32, 64):
    q_ = qkv.detach().requires_grad_()
    kv = q_[..., H * 128:]
    for name, ww in (("score", w), ("s0", None)):
        fn = lambda: PT.pool_gate_triton(kv, A, seg, ww, g, nw, 10.0, 1e-6, torch.bfloat16, bt)  # noqa: E731
        try:
            def fwd():
                with torch.no_grad():
                    fn()
            f_ms = bench(fwd)
            fb_ms = bench(lambda: fn().float().sum().backward())
            print(f"pool_gate_triton alone {name:5s} bt={bt} fwd {f_ms:6.2f} ms  bwd {fb_ms - f_ms:6.2f} ms  "
                  f"total {fb_ms:6.2f} ms", flush=True)
        except Exception as e:
            print(f"pool_gate_triton alone {name:5s} bt={bt} ERROR {type(e).__name__}: {e}", flush=True)
