"""Tests for the pooled gate (decay_pool) and gate modes. Run: .venv/bin/python tests/test_gates.py"""
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Train"))
sys.path.insert(0, str(ROOT / "tests"))
import model as M  # noqa: E402
import model_orig as MO  # noqa: E402

FAIL = []


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}", flush=True)
    if not ok:
        FAIL.append(name)


def naive(z, s, gamma, cu):
    """Per-token recurrence in float64: N = e^-g N + e^s z, D = e^-g D + e^s, reset at doc starts."""
    B, T, H, D = z.shape
    z, s = z.double(), (torch.zeros(B, T, H) if s is None else s).double()
    g = torch.zeros(H, dtype=torch.float64) if gamma is None else gamma.double()
    starts = set([0] if cu is None else cu[:-1].tolist())
    out = torch.zeros(B, T, H, D, dtype=torch.float64)
    for b in range(B):
        N = torch.zeros(H, D, dtype=torch.float64)
        Dn = torch.zeros(H, dtype=torch.float64)
        for t in range(T):
            if t in starts:
                N.zero_()
                Dn.zero_()
            w = torch.exp(s[b, t])
            N = torch.exp(-g)[:, None] * N + w[:, None] * z[b, t]
            Dn = torch.exp(-g) * Dn + w
            out[b, t] = N / Dn[:, None]
    return out


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp(min=1e-30)).item()


def rand_inputs(B, T, H, D, gen, dev):
    z = torch.randn(B, T, H, D, generator=gen).to(dev)
    s = (torch.rand(B, T, H, generator=gen) * 20 - 10).to(dev)
    return z, s


def test_pool_numeric(dev):
    gen = torch.Generator().manual_seed(0)
    H, D = 4, 8
    cu = torch.tensor([0, 1, 2, 63, 64, 65, 128, 200, 330, 331, 517])
    T = int(cu[-1])
    z, s = rand_inputs(1, T, H, D, gen, dev)
    gammas = {"mixed": torch.tensor([0.0, 1e-3, 0.25, 30.0]), "zero": None, "large": torch.tensor([80.0, 50.0, 10.0, 5.0])}
    for gname, g in gammas.items():
        for sname, ss in (("s", s), ("s0", None)):
            for C in (16, 64):
                out = M.decay_pool(z, ss, None if g is None else g.to(dev), cu.to(dev), C)
                ref = naive(z.cpu(), None if ss is None else ss.cpu(), g, cu)
                e = rel(out.cpu(), ref)
                check(f"match[{dev},gamma={gname},{sname},chunk={C},cu]", e < 1e-5 and out.isfinite().all().item(), f"rel_err={e:.2e}")
    # cu None, B > 1 (each row one doc)
    z, s = rand_inputs(3, 300, H, D, gen, dev)
    g = torch.tensor([0.0, 0.01, 0.3, 3.0])
    out = M.decay_pool(z, s, g.to(dev), None, 64)
    e = rel(out.cpu(), naive(z.cpu(), s.cpu(), g, None))
    check(f"match[{dev},cu=None,B=3]", e < 1e-5, f"rel_err={e:.2e}")
    # flattened B=1 with cu equals per-row B with cu=None when cu marks only row starts
    z2 = z.reshape(1, 900, H, D)
    s2 = s.reshape(1, 900, H)
    out2 = M.decay_pool(z2, s2, g.to(dev), torch.tensor([0, 300, 600, 900], device=dev), 64)
    e = rel(out2.reshape(3, 300, H, D), out)
    check(f"rows_vs_cu[{dev}]", e < 1e-6, f"rel_err={e:.2e}")


def test_causality_reset(dev):
    gen = torch.Generator().manual_seed(1)
    H, D = 3, 8
    cu = torch.tensor([0, 70, 71, 200, 333]).to(dev)
    T = 333
    z, s = rand_inputs(1, T, H, D, gen, dev)
    g = torch.tensor([0.0, 0.05, 2.0], device=dev)
    base = M.decay_pool(z, s, g, cu, 64)
    worst = 0.0
    for j in (5, 63, 64, 65, 150, 250, 332):
        z2, s2 = z.clone(), s.clone()
        z2[:, j] += 100.0
        s2[:, j] = 10.0
        out = M.decay_pool(z2, s2, g, cu, 64)
        worst = max(worst, (out[:, :j] - base[:, :j]).abs().max().item())
        changed = (out[:, j] - base[:, j]).abs().max().item()
        check(f"causality_changes_t=j[{dev},j={j}]", changed > 0)
    check(f"causality[{dev}]", worst == 0.0, f"max|dc_t<j|={worst:.1e}")
    # reset: c at a doc start equals z there; changing an earlier doc does not change a later doc
    for st in cu[:-1].tolist():
        e = (base[:, st] - z[:, st]).abs().max().item()
        check(f"reset_start_equals_z[{dev},t={st}]", e < 1e-5, f"err={e:.1e}")
    z2 = z.clone()
    z2[:, :200] = torch.randn(1, 200, H, D, generator=gen).to(dev)
    out = M.decay_pool(z2, s, g, cu, 64)
    e = (out[:, 200:] - base[:, 200:]).abs().max().item()
    check(f"reset_no_leak[{dev}]", e == 0.0, f"max_diff={e:.1e}")


def test_modes_closed_form(dev):
    gen = torch.Generator().manual_seed(2)
    H, D, T = 2, 4, 300
    z, s = rand_inputs(1, T, H, D, gen, dev)
    # gamma = 0: plain causal softmax over prefix
    w = torch.exp(s.double())
    ref = (w[..., None] * z.double()).cumsum(1) / w.cumsum(1)[..., None]
    e = rel(M.decay_pool(z, s, None, None, 64), ref)
    check(f"gamma0_is_prefix_softmax[{dev}]", e < 1e-5, f"rel_err={e:.2e}")
    # s = 0: exponentially weighted mean
    g = torch.tensor([0.1, 1.0], dtype=torch.float64)
    t = torch.arange(T, dtype=torch.float64)
    Wm = torch.exp(-g[:, None, None] * (t[:, None] - t[None, :])) * (t[:, None] >= t[None, :])
    ref = torch.einsum("hti,bihd->bthd", Wm, z.double().cpu()) / Wm.sum(-1).T[None, :, :, None]
    e = rel(M.decay_pool(z, None, g.float().to(dev), None, 64).cpu(), ref)
    check(f"s0_is_ewma[{dev}]", e < 1e-5, f"rel_err={e:.2e}")


def test_extremes_grad(dev):
    gen = torch.Generator().manual_seed(3)
    H, D, T = 4, 16, 16384
    z, _ = rand_inputs(1, T, H, D, gen, dev)
    for name, s in (("s=+10", torch.full((1, T, H), 10.0)), ("s=-10", torch.full((1, T, H), -10.0)),
                    ("s=rand", torch.rand(1, T, H, generator=gen) * 20 - 10),
                    ("s=alt", (torch.arange(T) % 2 * 20 - 10.0)[None, :, None].expand(1, T, H).contiguous())):
        for graw in (torch.tensor([-100.0, -20.0, 0.0, 100.0]), torch.tensor([-1e4, 1e-6, 30.0, 88.0])):
            zz = z.clone().requires_grad_()
            ss = s.to(dev).clone().requires_grad_()
            gr = graw.to(dev).clone().requires_grad_()
            out = M.decay_pool(zz, ss, torch.nn.functional.softplus(gr), None, 64)
            (out * torch.randn_like(out)).sum().backward()
            ok = out.isfinite().all() and zz.grad.isfinite().all() and ss.grad.isfinite().all() and gr.grad.isfinite().all()
            check(f"finite_long_doc[{dev},{name},gamma_raw={graw.tolist()}]", bool(ok))
    # gamma exactly 0 (arm C) over a 16k doc, plus the flattened multi-doc case
    zz = z.clone().requires_grad_()
    ss = (torch.rand(1, T, H, generator=gen) * 20 - 10).to(dev).requires_grad_()
    out = M.decay_pool(zz, ss, None, torch.tensor([0, 5, 9000, T], device=dev), 64)
    out.square().sum().backward()
    check(f"finite_gamma0_16k[{dev}]", bool(out.isfinite().all() and zz.grad.isfinite().all() and ss.grad.isfinite().all()))
    # long-doc numeric match vs reference (gamma 0 and small), chunk carry over 256 chunks
    zs, ss = z[:, :4096, :2, :4].contiguous(), (torch.rand(1, 4096, 2, generator=gen) * 20 - 10).to(dev)
    g = torch.tensor([0.0, 1e-3])
    e = rel(M.decay_pool(zs, ss, g.to(dev), None, 64).cpu(), naive(zs.cpu(), ss.cpu(), g, None))
    check(f"match_long_4096[{dev}]", e < 1e-5, f"rel_err={e:.2e}")


def cfg(mode, **kw):
    return M.ModelConfig(n_layers=kw.pop("n_layers", 12), pattern="G", gate_mode=mode, **kw)


def make_batch(dev, B=2, T=2048, seed=5):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(1, 32768, (B, T), generator=gen)
    x[0, 100] = 0
    x[1, 1500] = 0
    x[1, 1501] = 0
    y = torch.randint(0, 32768, (B, T), generator=gen)
    pos = torch.nonzero(x.reshape(-1) == 0).flatten() + 1
    cu = torch.unique(torch.cat([torch.arange(0, B * T + 1, T), pos]))
    return x.to(dev), y.to(dev), cu.to(dev), cu


def test_arm_a_vs_orig(dev):
    torch.manual_seed(1234)
    orig = MO.LM(MO.ModelConfig(n_layers=12, pattern="G")).to(dev)
    torch.manual_seed(1234)
    new = M.LM(cfg("linear")).to(dev)
    so, sn = orig.state_dict(), new.state_dict()
    same = so.keys() == sn.keys() and all(torch.equal(so[k], sn[k]) for k in so)
    check("armA_state_dict_equals_orig_same_seed", same)
    x, y, cu, cu_cpu = make_batch(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = orig(x, y, cu, cu_cpu)
        ln = new(x, y, cu, cu_cpu)
        go = orig(x)
        gn = new(x)
    check("armA_loss_equals_orig", torch.equal(lo, ln), f"{lo.item():.6f} vs {ln.item():.6f}")
    check("armA_logits_equal_orig(no cu)", torch.equal(go, gn))
    # single layer with copied weights
    torch.manual_seed(7)
    lo_ = MO.GatedDeltaNet(MO.ModelConfig()).to(dev)
    ln_ = M.GatedDeltaNet(cfg("linear")).to(dev)
    ln_.load_state_dict(lo_.state_dict())
    h = torch.randn(1, 4096, 768, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cu2 = torch.tensor([0, 1000, 1001, 4096], device=dev)
        a, b = lo_(h, cu2, cu2.cpu()), ln_(h, cu2, cu2.cpu())
    check("armA_layer_equals_orig_layer", torch.equal(a, b))


def test_init_equal(dev):
    x, y, cu, cu_cpu = make_batch(dev)
    torch.manual_seed(1234)
    A = M.LM(cfg("linear")).to(dev)
    sa = A.state_dict()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        la = A(x, y, cu, cu_cpu)
    for mode in ("pool", "pool_g0", "pool_s0"):
        torch.manual_seed(1234)
        Mo = M.LM(cfg(mode)).to(dev)
        sm = Mo.state_dict()
        shared = all(torch.equal(sa[k], sm[k]) for k in sa)
        extra = sorted({k.split(".")[-1] if "c_norm" not in k else "c_norm.weight" for k in sm if k not in sa})
        check(f"init_shared_params_equal[{mode}]", shared, f"extra={extra}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lm = Mo(x, y, cu, cu_cpu)
        check(f"init_loss_equals_A[{mode}]", torch.equal(la, lm), f"{la.item():.6f} vs {lm.item():.6f}")
        n_extra = Mo.num_params() - A.num_params()
        print(f"     params[{mode}] = {Mo.num_params()} (+{n_extra} vs A {A.num_params()})")
        # no-decay group holds score_w and gamma_raw
        groups = Mo.param_groups(0.1)
        nd = {id(p) for p in groups[1]["params"]}
        names = [n for n, p in Mo.named_parameters() if n.endswith((".score_w", ".gamma_raw"))]
        check(f"no_decay_score_gamma[{mode}]", all(id(dict(Mo.named_parameters())[n]) in nd for n in names), f"n={len(names)}")
        if mode == "pool":
            g = Mo.blocks[0].mixer.gamma().double()
            hz = (1 / -torch.expm1(-g)).tolist()
            check("gamma_init_horizons", abs(hz[0] - 4) < 1e-3 and abs(hz[-1] - 256) < 1e-2, f"{[round(v, 2) for v in hz]}")


def test_grads(dev):
    x, y, cu, cu_cpu = make_batch(dev)
    for mode in ("pool", "pool_g0", "pool_s0"):
        res = {}
        for ckpt in (True, False):
            torch.manual_seed(1234)
            m = M.LM(cfg(mode, n_layers=2, pool_ckpt=ckpt)).to(dev)
            with torch.no_grad():
                for blk in m.blocks:
                    blk.mixer.ab_c.normal_(0, 0.02, generator=torch.Generator(device=dev).manual_seed(9))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x, y, cu, cu_cpu)
            loss.backward()
            res[ckpt] = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
            fin = all(gr.isfinite().all() for gr in res[ckpt].values())
            check(f"grads_finite[{mode},ckpt={ckpt}]", fin and len(res[ckpt]) == len(list(m.parameters())), f"n_grads={len(res[ckpt])}")
            mx = m.blocks[0].mixer
            nz = {n: res[ckpt][f"blocks.0.mixer.{n}"].abs().sum().item() for n in ("score_w", "gamma_raw", "c_proj", "c_norm.weight", "ab_c") if f"blocks.0.mixer.{n}" in res[ckpt]}
            check(f"grads_nonzero_pool_params[{mode},ckpt={ckpt}]", all(v > 0 for v in nz.values()), str({k: f'{v:.2e}' for k, v in nz.items()}))
        e = max((res[True][n] - res[False][n]).abs().max().item() for n in res[True])
        check(f"ckpt_vs_nockpt_grads[{mode}]", e < 1e-5, f"max_diff={e:.1e}")
        # gradient reaches qkv (k, v) through c: differs from a run with c-path detached
        torch.manual_seed(1234)
        m = M.LM(cfg(mode, n_layers=1, pool_ckpt=False)).to(dev)
        mx = m.blocks[0].mixer
        with torch.no_grad():
            mx.ab_c.normal_(0, 0.02, generator=torch.Generator(device=dev).manual_seed(9))
        gq = []
        for det in (False, True):
            m.zero_grad()
            orig_pooled = mx.pooled
            if det:
                mx.pooled = lambda kv, cu: orig_pooled(kv.detach(), cu)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(x, y, cu, cu_cpu).backward()
            mx.pooled = orig_pooled
            gq.append(mx.qkv.weight.grad.clone())
        d = (gq[0] - gq[1]).abs().max().item()
        check(f"grad_flows_through_c_into_kv[{mode}]", d > 0, f"max|dgrad_qkv|={d:.2e}")


if __name__ == "__main__":
    dev = torch.device("cuda", 0)
    for d in ("cpu", dev):
        test_pool_numeric(d)
        test_causality_reset(d)
        test_modes_closed_form(d)
    test_extremes_grad(dev)
    test_arm_a_vs_orig(dev)
    test_init_equal(dev)
    test_grads(dev)
    print(f"\n{len(FAIL)} FAILED: {FAIL}" if FAIL else "\nALL PASSED")
    sys.exit(1 if FAIL else 0)
