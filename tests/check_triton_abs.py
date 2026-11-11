"""Max-abs-diff report: pool_gate_triton (bt=32, 64) vs model.pool_gate (float32 out), out + grads.
Run: HIP_VISIBLE_DEVICES=0 .venv/bin/python tests/check_triton_abs.py"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Train"))
import model as M  # noqa: E402
from pool_triton import pool_gate_triton  # noqa: E402

dev = torch.device("cuda", 0)


def run(impl, bt, k, v, cu, w, graw, nw, B, T, dout):
    k, v, nw = (t.detach().clone().requires_grad_() for t in (k, v, nw))
    w = None if w is None else w.detach().clone().requires_grad_()
    graw = None if graw is None else graw.detach().clone().requires_grad_()
    H = k.shape[2]
    if impl == "triton":
        seg = M.segment_ids(B, T, cu, dev)
        if B > 1:
            seg = seg + torch.arange(B, device=dev)[:, None] * (T + 1)
        g = F.softplus(graw) if graw is not None else torch.zeros(H, device=dev)
        out = pool_gate_triton(k, v, seg, w, g, nw, 10.0, 1e-6, torch.float32, bt)
    else:
        seg = M.padded_segments(B, T, cu, 64, dev)
        out, _ = M.pool_gate(k, v, seg, w, graw, nw, 64, 10.0, 1e-6)
    (out * dout).sum().backward()
    gr = {"k": k.grad, "v": v.grad, "norm_w": nw.grad}
    if w is not None:
        gr["score_w"] = w.grad
    if graw is not None:
        gr["gamma_raw"] = graw.grad
    return out, gr


def case(name, B, T, cu, score=True, gamma=True, H=6, DK=128):
    gen = torch.Generator().manual_seed(7)
    k = torch.randn(B, T, H, DK, generator=gen).to(dev)
    v = torch.randn(B, T, H, DK, generator=gen).to(dev)
    w = (torch.randn(H, 2 * DK, generator=gen) * 0.2).to(dev) if score else None
    graw = torch.linspace(-6, 3, H).to(dev) if gamma else None
    nw = (1 + 0.1 * torch.randn(2 * DK, generator=gen)).to(dev)
    dout = torch.randn(B, T, H * 2 * DK, generator=gen).to(dev)
    o_ref, g_ref = run("torch", None, k, v, cu, w, graw, nw, B, T, dout)
    for bt in (32, 64):
        try:
            o, g = run("triton", bt, k, v, cu, w, graw, nw, B, T, dout)
        except Exception as e:  # report compile errors verbatim
            print(f"{name:28s} bt={bt} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        parts = [f"out {(o - o_ref).abs().max().item():.2e} (|ref|max {o_ref.abs().max().item():.1f})"]
        for n in g_ref:
            d = (g[n].float() - g_ref[n].float()).abs().max().item()
            parts.append(f"{n} {d:.2e}/{g_ref[n].abs().max().item():.1e}")
        fin = o.isfinite().all().item() and all(x.isfinite().all().item() for x in g.values())
        print(f"{name:28s} bt={bt} finite={fin} " + "  ".join(parts), flush=True)


if __name__ == "__main__":
    print("max abs diff triton - torch (grad: absdiff/|ref|max)")
    T = 4096
    cu = torch.tensor([0, 1, 2, 63, 64, 65, 100, 128, 700, 2047, 2048, 3000, 3001, T], device=dev)
    case("B pool cu", 1, T, cu)
    case("C pool_g0 (gamma 0) cu", 1, T, cu, gamma=False)
    case("D pool_s0 (score None) cu", 1, T, cu, score=False)
    case("B no_cu B=3 T=1000", 3, 1000, None)
    case("B T=517 pad", 1, 517, torch.tensor([0, 100, 517], device=dev))
    T = 32768
    torch.manual_seed(0)
    cu = torch.unique(torch.cat([torch.arange(0, T + 1, 2048), torch.randint(1, T, (12,))])).to(dev)
    case("B T=32768 bench cu", 1, T, cu)
