"""Which pool_triton kernel fails at bt=64: launch each kernel alone and report shared-mem / errors.
Run: HIP_VISIBLE_DEVICES=0 .venv/bin/python tests/probe_triton_bt64.py"""
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Train"))
import pool_triton as PT  # noqa: E402

dev = torch.device("cuda", 0)
N, H, DK = 4096, 6, 128
DP = 2 * DK + 1
k = torch.randn(N, H, DK, device=dev, dtype=torch.bfloat16)
v = torch.randn(N, H, DK, device=dev, dtype=torch.bfloat16)
seg = torch.zeros(N, device=dev, dtype=torch.int32)
w = torch.randn(H, 2 * DK, device=dev) * 0.02
g = torch.full((H,), 0.05, device=dev)
nw = torch.ones(2 * DK, device=dev)
st, sh = k.stride(0), k.stride(1)
for bt in (32, 64):
    nc = triton.cdiv(N, bt)
    E = torch.zeros(nc, H, DP, device=dev)
    DEC = torch.zeros(nc, H, device=dev)
    S = torch.zeros(nc, H, DP, device=dev)
    out = torch.empty(N, H, 2 * DK, device=dev, dtype=torch.bfloat16)
    do = torch.randn(N, H, 2 * DK, device=dev)
    FX = torch.zeros(nc, H, DP, device=dev)
    dk = torch.empty(N, H, DK, device=dev, dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    DW = torch.empty(nc, H, 2 * DK, device=dev)
    DG = torch.empty(nc, H, device=dev)
    DNW = torch.empty(nc, H, 2 * DK, device=dev)
    for hs in (True, False):
        launches = {
            "_p1_chunk_end": lambda: PT._p1_chunk_end[(nc, H)](k, v, seg, w, g, E, DEC, N, st, sh, 10.0, H=H, BT=bt, DK=DK,
                                                               DP=DP, HAS_SCORE=hs, num_warps=PT.NUM_WARPS),
            "_p3_out": lambda: PT._p3_out[(nc, H)](k, v, seg, w, g, S, nw, out, N, st, sh, 10.0, 1e-6, H=H, BT=bt, DK=DK,
                                                   DP=DP, HAS_SCORE=hs, num_warps=PT.NUM_WARPS),
            "_b1": lambda: PT._b1[(nc, H)](k, v, seg, w, g, S, nw, do, FX, N, st, sh, 10.0, 1e-6, H=H, BT=bt, DK=DK,
                                           DP=DP, HAS_SCORE=hs, num_warps=PT.NUM_WARPS),
            "_b3": lambda: PT._b3[(nc, H)](k, v, seg, w, g, S, nw, do, FX, dk, dv, DW, DG, DNW, N, st, sh, 10.0, 1e-6,
                                           H=H, BT=bt, DK=DK, DP=DP, HAS_SCORE=hs, num_warps=PT.NUM_WARPS),
        }
        for name, fn in launches.items():
            try:
                kern = fn()
                torch.cuda.synchronize()
                print(f"bt={bt} HAS_SCORE={hs!s:5s} {name:14s} ok  shared={kern.metadata.shared} B "
                      f"regs={getattr(kern, 'n_regs', '?')} spills={getattr(kern, 'n_spills', '?')}", flush=True)
            except Exception as e:
                print(f"bt={bt} HAS_SCORE={hs!s:5s} {name:14s} ERROR {type(e).__name__}: {e}", flush=True)
