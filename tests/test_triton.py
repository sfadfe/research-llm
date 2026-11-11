"""Fused Triton pooled gate vs the torch reference path (model.pool_gate, itself tested against the naive loop).
Run: .venv/bin/python tests/test_triton.py"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Train"))
sys.path.insert(0, str(ROOT / "tests"))
import model as M  # noqa: E402
from pool_triton import pool_core_triton, pool_gate_triton  # noqa: E402
from test_gates import naive  # noqa: E402

FAIL = []
dev = torch.device("cuda", 0)


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}", flush=True)
    if not ok:
        FAIL.append(name)


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp(min=1e-30)).item()


def segs(B, T, cu):
    seg = M.segment_ids(B, T, cu, dev)
    if B > 1:
        seg = seg + torch.arange(B, device=dev)[:, None] * (T + 1)
    return seg


def run(impl, qkv, cu, A, w, graw, nw, B, T, dt=torch.float32):
    """qkv (B,T,3*H*DK) leaf; kv = qkv[..., H*DK:] as in the model. Grads: k, v (slices of qkv.grad), A, w, gamma, nw."""
    H = A.shape[0] // nw.shape[0]
    qkv = qkv.detach().clone().requires_grad_()
    A = A.detach().clone().requires_grad_()
    w = None if w is None else w.detach().clone().requires_grad_()
    graw = None if graw is None else graw.detach().clone().requires_grad_()
    nw = nw.detach().clone().requires_grad_()
    kd = qkv.shape[-1] // 3
    kv = qkv[..., kd:]
    g = F.softplus(graw) if graw is not None else torch.zeros(H, device=dev)
    if impl == "triton":
        out = pool_gate_triton(kv, A, segs(B, T, cu), w, g, nw, 10.0, 1e-6, dt)
    else:
        k, v = kv.unflatten(-1, (2, H, kd // H)).unbind(-3)
        out, _ = M.pool_gate(k, v, M.padded_segments(B, T, cu, 64, dev), A, w, graw, nw, 64, 10.0, 1e-6)
    gen = torch.Generator(device=dev).manual_seed(11)
    dout = torch.randn(out.shape, device=dev, generator=gen)
    (out.float() * dout).sum().backward()
    grads = {"k": qkv.grad[..., kd:2 * kd], "v": qkv.grad[..., 2 * kd:], "A": A.grad, "nw": nw.grad}
    if w is not None:
        grads["w"] = w.grad
    if graw is not None:
        grads["gamma_raw"] = graw.grad
    return out, grads


def make(gen, B, T, H, DK, R, score, kdt=torch.float32):
    qkv = torch.randn(B, T, 3 * H * DK, generator=gen).to(dev, kdt)
    A = (torch.randn(H * R, 2 * DK, generator=gen) / (2 * DK) ** 0.5).to(dev)
    w = (torch.randn(H, 2 * DK, generator=gen) * 0.2).to(dev) if score else None
    nw = (1 + 0.1 * torch.randn(R, generator=gen)).to(dev)
    return qkv, A, w, nw


def compare(name, B, T, cu, H=6, DK=128, R=16, score=True, gamma=True, graw_val=None, kdt=torch.float32, tol=2e-4,
            gtol=2e-3, kvtol=None):
    gen = torch.Generator().manual_seed(hash(name) % 1000)
    qkv, A, w, nw = make(gen, B, T, H, DK, R, score, kdt)
    if gamma:
        graw = (torch.linspace(-6, 3, H) if graw_val is None else graw_val).to(dev)
    else:
        graw = None
    o1, g1 = run("triton", qkv, cu, A, w, graw, nw, B, T)
    o2, g2 = run("torch", qkv, cu, A, w, graw, nw, B, T)
    e = rel(o1, o2)
    ok = e < tol and o1.isfinite().all().item()
    ge = {n: rel(g1[n], g2[n]) for n in g2}
    kvtol = gtol if kvtol is None else kvtol
    okg = all(v_ < (kvtol if n in ("k", "v") else gtol) for n, v_ in ge.items())
    okg = okg and all(g1[n].isfinite().all().item() for n in g1)
    check(f"triton_vs_torch_out[{name}]", ok, f"rel_err={e:.2e}")
    check(f"triton_vs_torch_grad[{name}]", okg, " ".join(f"{n}={x:.1e}" for n, x in ge.items()))


def core_ref(k, v, x, seg, w, graw, nw, H):
    """float32 torch reference of the pooling core on a given x (score from [k; v], as in model.pool_gate)."""
    z = torch.cat([k, v], -1).float()
    B, T = z.shape[:2]
    s = 10.0 * torch.tanh(torch.einsum("bthd,hd->bth", z, w) / 10.0) if w is not None else torch.zeros(B, T, H, device=dev)
    g = F.softplus(graw) if graw is not None else torch.zeros(H, device=dev)
    c = M.pool_chunks(x.float(), s, g, seg, 64)
    return F.rms_norm(c, (x.shape[-1],), nw, 1e-6).flatten(-2)


def compare_bf16_io(name, B, T, cu, H=6, DK=128, R=16, score=True, gamma=True):
    """Pooling core (pool_core_triton) with bf16 k, v, x in / bf16 out / bf16 dout vs the float32 torch reference fed
    the same bf16 k, v, x, dout. Only the bf16 stores of out, dx, dk, dv may differ (<= 1 bf16 rounding, 2^-8 rel)."""
    gen = torch.Generator().manual_seed(17)
    qkv, A, w, nw = make(gen, B, T, H, DK, R, score, torch.bfloat16)
    kd = H * DK
    x = torch.randn(B, T, H, R, generator=gen).to(dev, torch.bfloat16)
    graw = torch.linspace(-6, 3, H).to(dev) if gamma else None
    dout = torch.randn(B, T, H * R, generator=gen).to(dev, torch.bfloat16)
    leaves = lambda *ts: [None if t is None else t.detach().clone().requires_grad_() for t in ts]  # noqa: E731
    q1, x1, w1, gr1, nw1 = leaves(qkv, x, w, graw, nw)
    k1, v1 = q1[..., kd:].unflatten(-1, (2, H, DK)).unbind(-3)
    g1 = F.softplus(gr1) if gr1 is not None else torch.zeros(H, device=dev)
    o1 = pool_core_triton(k1, v1, x1, segs(B, T, cu), w1, g1, nw1, 10.0, 1e-6, torch.bfloat16)
    o1.backward(dout)
    q2, x2, w2, gr2, nw2 = leaves(qkv.float(), x.float(), w, graw, nw)
    k2, v2 = q2[..., kd:].unflatten(-1, (2, H, DK)).unbind(-3)
    o2 = core_ref(k2, v2, x2, M.padded_segments(B, T, cu, 64, dev), w2, gr2, nw2, H)
    o2.backward(dout.float())
    ok_dt = o1.dtype == x1.grad.dtype == torch.bfloat16 and (not score or q1.grad.dtype == torch.bfloat16)

    def ew(a, b):  # worst elementwise error in units of (2^-8 |ref| + 1e-5 max|ref|)
        a, b = a.double(), b.double()
        return ((a - b).abs() / (2 ** -8 * b.abs() + 1e-5 * b.abs().max())).max().item()

    e = {"out": ew(o1, o2), "x": ew(x1.grad, x2.grad)}
    if score:
        e["k"] = ew(q1.grad[..., kd:2 * kd], q2.grad[..., kd:2 * kd])
        e["v"] = ew(q1.grad[..., 2 * kd:], q2.grad[..., 2 * kd:])
    r = {"norm_w": rel(nw1.grad, nw2.grad)}
    if w is not None:
        r["score_w"] = rel(w1.grad, w2.grad)
    if graw is not None:
        r["gamma_raw"] = rel(gr1.grad, gr2.grad)
    fin = o1.isfinite().all().item() and x1.grad.isfinite().all().item() and (not score or q1.grad.isfinite().all().item())
    # 1 unit = max bf16 rounding error; 5% slack for fp32 summation-order differences at rounding midpoints
    ok = ok_dt and fin and all(x_ <= 1.05 for x_ in e.values()) and all(x_ < 1e-4 for x_ in r.values())
    check(f"triton_bf16_io_vs_fp32_ref[{name}]", ok,
          " ".join(f"{n}={x_:.2f}ulp" for n, x_ in e.items()) + " " + " ".join(f"{n}={x_:.1e}" for n, x_ in r.items()))


def compare_bf16_e2e(name, B, T, cu, H=6, DK=128, R=16, score=True, gamma=True):
    """Full pool_gate_triton in bf16 (x = A [k; v] rounded to bf16 once, out bf16) vs the float32 torch reference fed
    the same bf16 k, v (x not rounded). Error budget is set by the bf16 rounding of x (2^-9 rel) and out."""
    gen = torch.Generator().manual_seed(23)
    qkv, A, w, nw = make(gen, B, T, H, DK, R, score, torch.bfloat16)
    graw = torch.linspace(-6, 3, H).to(dev) if gamma else None
    o1, g1 = run("triton", qkv, cu, A, w, graw, nw, B, T, torch.bfloat16)
    o2, g2 = run("torch", qkv.float(), cu, A, w, graw, nw, B, T)
    e = rel(o1, o2)
    ge = {n: rel(g1[n], g2[n]) for n in g2}
    ok = o1.dtype == torch.bfloat16 and e < 2e-2 and all(x_ < 3e-2 for x_ in ge.values())
    check(f"triton_bf16_e2e_vs_fp32_ref[{name}]", ok, f"out={e:.1e} " + " ".join(f"{n}={x_:.1e}" for n, x_ in ge.items()))


if __name__ == "__main__":
    T = 4096
    cu = torch.tensor([0, 1, 2, 63, 64, 65, 128, 700, 2047, 2048, 3000, 3001, T], device=dev)
    compare("B_mode_cu", 1, T, cu)
    compare("C_mode_gamma0", 1, T, cu, gamma=False)
    compare("D_mode_s0", 1, T, cu, score=False)
    compare("no_cu_B3_T1000", 3, 1000, None)
    compare("pad_T=517", 1, 517, torch.tensor([0, 100, 517], device=dev))
    compare("gamma_extreme", 1, T, cu, graw_val=torch.tensor([-30.0, -10.0, 0.0, 10.0, 50.0, 88.0]))
    # bf16 k, v leaves: triton dk/dv = bf16 sum of two bf16 grads (projection path + score path), ref one rounding
    compare("bf16_kv", 1, T, cu, kdt=torch.bfloat16, kvtol=1e-2)
    compare("long_doc_gamma0_16k", 1, 16384, torch.tensor([0, 16384], device=dev), gamma=False)
    # bf16 I/O (as used under autocast): all modes, unaligned cu, B>1 without cu, T not a multiple of the chunk
    compare_bf16_io("B_mode_cu", 1, T, cu)
    compare_bf16_io("C_mode_gamma0", 1, T, cu, gamma=False)
    compare_bf16_io("D_mode_s0", 1, T, cu, score=False)
    compare_bf16_io("B_no_cu_B3_T1000", 3, 1000, None)
    compare_bf16_io("C_no_cu_B2_T777", 2, 777, None, gamma=False)
    compare_bf16_io("D_pad_T=517", 1, 517, torch.tensor([0, 100, 333, 517], device=dev), score=False)
    compare_bf16_e2e("B_mode_cu", 1, T, cu)
    compare_bf16_e2e("C_mode_gamma0", 1, T, cu, gamma=False)
    compare_bf16_e2e("D_mode_s0", 1, T, cu, score=False)
    # bf16 output equals fp32 output rounded to bf16 (same bf16 x)
    gen = torch.Generator().manual_seed(5)
    k = torch.randn(1, 2048, 6, 128, generator=gen).to(dev, torch.bfloat16)
    v = torch.randn(1, 2048, 6, 128, generator=gen).to(dev, torch.bfloat16)
    x = torch.randn(1, 2048, 6, 16, generator=gen).to(dev, torch.bfloat16)
    seg = M.segment_ids(1, 2048, torch.tensor([0, 500, 2048], device=dev), dev)
    w = torch.randn(6, 256, device=dev) * 0.02
    g = torch.full((6,), 0.1, device=dev)
    nw = torch.ones(16, device=dev)
    a = pool_core_triton(k, v, x, seg, w, g, nw, 10.0, 1e-6, torch.float32)
    b = pool_core_triton(k, v, x, seg, w, g, nw, 10.0, 1e-6, torch.bfloat16)
    check("bf16_out_is_rounded_fp32", torch.equal(a.bfloat16(), b))
    # whole-model: triton vs torch impl, same weights, loss and grads
    from test_gates import make_batch
    x, y, cu, cu_cpu = make_batch(dev)
    for mode in ("pool", "pool_g0", "pool_s0"):
        res = {}
        for impl in ("triton", "torch"):
            torch.manual_seed(1234)
            m = M.LM(M.ModelConfig(n_layers=2, pattern="G", gate_mode=mode, pool_impl=impl)).to(dev)
            with torch.no_grad():
                for blk in m.blocks:
                    blk.mixer.ab_c.normal_(0, 0.02, generator=torch.Generator(device=dev).manual_seed(9))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x, y, cu, cu_cpu)
            loss.backward()
            res[impl] = (loss.item(), {n: p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
        dl = abs(res["triton"][0] - res["torch"][0])
        ge = max(rel(res["triton"][1][n], res["torch"][1][n]) for n in res["torch"][1])
        worst = max(res["torch"][1], key=lambda n: rel(res["triton"][1][n], res["torch"][1][n]))
        check(f"model_triton_vs_torch[{mode}]", dl < 2e-3 and ge < 5e-2, f"dloss={dl:.1e} max_grad_rel={ge:.1e} ({worst})")
    print(f"\n{len(FAIL)} FAILED: {FAIL}" if FAIL else "\nALL PASSED")
    sys.exit(1 if FAIL else 0)
