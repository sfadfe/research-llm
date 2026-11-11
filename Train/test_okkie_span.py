"""Tests for gate_mode = "span" (Train/okkie_span.py). GPU (fla kernels), small models.
Usage (from research-llm): .venv/bin/python Train/test_okkie_span.py [--ref REF.pt] [--speed] [--smoke DIR]
                           [--device cuda:1]
  --ref: logits of the 117M plain / okkie-transformer-117M checkpoints saved BEFORE the model.py edit
         (same input as ref_logits below); compared bit for bit.
  --speed: forward+backward tokens/s and peak memory of the four run configs at their micro_batch (cu = None, as
           doc_mask = false), spans per sequence on data/switch at random init.
  --smoke DIR: a few steps of Train/train.py for each of the four configs on --device (scratch run dirs in DIR)."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(os.path.abspath(__file__)).parent  # not resolved: a symlinked copy uses its own dir
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from model import LM, ModelConfig  # noqa: E402
import okkie_span  # noqa: E402
from okkie_span import read_masks, span_ids, span_read, l2n, starts_to_ids, ema  # noqa: E402

DEV = "cuda:0"
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)


def small_cfg(mode, **kw):
    c = dict(vocab_size=1000, d_model=128, n_layers=2, pattern="G", mlp_hidden=256, gdn_heads=2, gdn_head_dim=64,
             loss="plain", gate_mode=mode, pool_chunk=64, span_hidden=64, span_dim=16, mung_dim=32, span_max=8)
    c.update(kw)
    return ModelConfig(**c)


def build(mode, seed=0, **kw):
    torch.manual_seed(seed)
    return LM(small_cfg(mode, **kw)).to(DEV)


def spans_of(model):
    return [b.mixer.span for b in model.blocks]


def tune_theta(model, x, q=0.4, **kw):
    """Per layer: set theta to the q-quantile of that layer's s so that several spans occur."""
    xs = []
    hooks = [b.mixer.register_forward_hook(lambda m, i, o: xs.append(i[0])) for b in model.blocks]
    with torch.no_grad():
        model(x, **kw)
    for h in hooks:
        h.remove()
    for sp, xi in zip(spans_of(model), xs):
        z = sp.features(xi).float()
        _, s = span_ids(z, sp.h_short, sp.h_long, 0.0, sp.S)  # with cu: flattened, approximate quantile
        sp.theta = float(torch.quantile(s[:, 1:].flatten(), q))


def perturb(model, gen):
    with torch.no_grad():
        for sp in spans_of(model):
            sp.tau.copy_(torch.randn(sp.tau.shape, generator=gen) * 2)
            sp.gate_b.copy_(torch.randn(sp.gate_b.shape, generator=gen))
            sp.rho.copy_(torch.randn(sp.rho.shape, generator=gen))
            sp.read_log_scale.add_(torch.randn(sp.read_log_scale.shape, generator=gen).to(DEV) * 0.3)


def record(model, on=True):
    for sp in spans_of(model):
        sp.record = on


def n_spans(model):
    out = []
    for sp in spans_of(model):
        sid = sp.last["sid"]
        if sp.last["valid"] is not None:
            sid = sid.masked_fill(~sp.last["valid"], 0)
        out.append((sid.max(1).values + 1).float().mean().item())
    return out


# ---------------------------------------------------------------------------------------------- causality
def sids(model):
    """Per layer span ids in the per-document layout, pads set to -1."""
    out = []
    for sp in spans_of(model):
        sid = sp.last["sid"]
        out.append(sid if sp.last["valid"] is None else sid.masked_fill(~sp.last["valid"], -1))
    return out


def test_causality(tune=True, tag="", **cfg):
    """tune: theta at the 0.4-quantile of s (many threshold spans). tune False: default theta (few threshold
    crossings at init), so the spans come from the forced boundaries (span_max_len). Also checks that the span ids
    of tokens before t do not change."""
    gen = torch.Generator().manual_seed(1)
    m = build("span", **cfg).eval()
    perturb(m, gen)
    T = 256
    x = torch.randint(0, 1000, (2, T), generator=gen).to(DEV)
    if tune:
        tune_theta(m, x)
    record(m)
    with torch.no_grad():
        y0 = m(x)
        ns = n_spans(m)
        s0 = sids(m)
        nread = [(sp.last["high"].any(-1)).float().mean().item() for sp in spans_of(m)]
        for t in (37, 130, 200):
            x1 = x.clone()
            x1[:, t] = (x1[:, t] + 1 + torch.randint(0, 998, (2,), generator=gen).to(DEV)) % 1000
            y1 = m(x1)
            sid_ok = all(torch.equal(a[:, :t], b[:, :t]) for a, b in zip(s0, sids(m)))
            d_before = (y1[:, :t] - y0[:, :t]).abs().max().item()
            d_after = (y1[:, t:] - y0[:, t:]).abs().max().item()
            check(f"causality{tag} cu=None t={t}", d_before < 1e-4 and d_after > 1e-3 and sid_ok,
                  f"max|d| before t {d_before:.2e}, at/after t {d_after:.2e}; span ids before t same {sid_ok}; "
                  f"spans/seq per layer {ns}, frac tokens with a high span {[round(v, 3) for v in nread]}")
    # flattened with cu: unequal documents (gather layout); changes stay inside the document and after t
    cu = torch.tensor([0, 90, 256, 300, 512])
    kw = dict(cu=cu.to(DEV), cu_cpu=cu)
    if tune:
        tune_theta(m, x, **kw)
    doc = torch.repeat_interleave(torch.arange(4), cu[1:] - cu[:-1])
    pos = torch.arange(512) - cu[doc]
    with torch.no_grad():
        y0 = m(x, **kw)[0]
        ns = n_spans(m)
        s0 = [s.cpu()[doc, pos] for s in sids(m)]  # flat (512,) span id of each token
        for t, end in ((40, 90), (150, 256), (270, 300)):
            x1 = x.clone().view(-1)
            x1[t] = (x1[t] + 7) % 1000
            y1 = m(x1.view(2, T), **kw)[0]
            s1 = [s.cpu()[doc, pos] for s in sids(m)]
            sid_ok = all(torch.equal(a[:t], b[:t]) and torch.equal(a[end:], b[end:]) for a, b in zip(s0, s1))
            d_before = (y1[:t] - y0[:t]).abs().max().item()
            d_other = (y1[end:] - y0[end:]).abs().max().item()
            d_in = (y1[t:end] - y0[t:end]).abs().max().item()
            check(f"causality{tag}+doc isolation cu t={t}",
                  d_before < 1e-4 and d_other < 1e-4 and d_in > 1e-3 and sid_ok,
                  f"before {d_before:.2e}, later docs {d_other:.2e}, own doc after t {d_in:.2e}; span ids before "
                  f"t / later docs same {sid_ok}; spans/doc {ns}")
    record(m, False)


# ---------------------------------------------------------------------------------------------- forced boundaries
def ref_ids(start, max_len):
    """Loop reference: a span starts at a threshold start or max_len tokens after the last start."""
    N, L = start.shape
    out = torch.zeros(N, L, dtype=torch.long)
    for n in range(N):
        sid, last = 0, 0
        for t in range(1, L):
            if bool(start[n, t]) or (max_len and t - last >= max_len):
                sid, last = sid + 1, t
            out[n, t] = sid
    return out


def run_lengths(sid):
    """sid (L,) nondecreasing, pads -1 -> lengths of each id present."""
    sid = sid[sid >= 0]
    return torch.unique_consecutive(sid, return_counts=True)[1]


def test_forced_ids():
    gen = torch.Generator().manual_seed(6)
    ok, n = True, 0
    for L in (1, 2, 5, 64, 300):
        for p in (0.0, 0.02, 0.2):
            start = torch.rand(3, L, generator=gen) < p
            start[:, 0] = False  # as span_ids builds it
            for ml in (0, None, 1, 3, 16, 128):
                got = starts_to_ids(start.to(DEV), ml).cpu()
                ok &= torch.equal(got, ref_ids(start, ml))
                n += 1
    check("starts_to_ids == loop reference (threshold or max_len after last start)", ok, f"{n} cases")
    # causal: id_t depends on start[:, :t+1] only
    start = torch.rand(4, 500, generator=gen) < 0.03
    start[:, 0] = False
    base = starts_to_ids(start.to(DEV), 16).cpu()
    ok = True
    for t in (0, 10, 200, 499):
        s2 = start.clone()
        s2[:, t + 1:] = torch.rand(4, 500 - t - 1, generator=gen) < 0.5
        ok &= torch.equal(starts_to_ids(s2.to(DEV), 16).cpu()[:, :t + 1], base[:, :t + 1])
    check("starts_to_ids causal (changing starts after t keeps ids <= t)", ok)
    # span_ids with constant z (s = 1, no threshold start): ids = t // max_len, clamped
    z = torch.ones(2, 300, 4, device=DEV)
    sid, _ = span_ids(z, 4.0, 64.0, 0.5, 32, max_len=128)
    want = (torch.arange(300, device=DEV) // 128).expand(2, -1)
    check("span_ids: no threshold start -> ids t // span_max_len", torch.equal(sid, want))
    sid, _ = span_ids(z, 4.0, 64.0, 0.5, 2, max_len=128)
    check("span_ids: forced ids still clamped to span_max - 1", int(sid.max()) == 1)
    sid1, s1 = span_ids(torch.randn(3, 1, 4, device=DEV), 4.0, 64.0, 0.5, 32, max_len=128)
    sid0, _ = span_ids(torch.randn(3, 1, 4, device=DEV), 4.0, 64.0, 0.5, 32)
    check("span_ids L=1 -> (N,1) zeros (forced on / off)", sid1.shape == (3, 1) and sid0.shape == (3, 1)
          and not sid1.any() and not sid0.any(), f"{tuple(sid1.shape)} {tuple(sid0.shape)}")


def old_span_ids(z, h_short, h_long, theta, span_max, valid=None, max_len=None):
    """span_ids before span_max_len (max_len ignored)."""
    with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
        z = z.float()
        if valid is not None:
            z = z * valid[..., None]
        s = F.cosine_similarity(ema(z, h_short), ema(z, h_long), dim=-1, eps=1e-8)
        start = (s[:, 1:] < theta) & (s[:, :-1] >= theta)
        start = torch.cat([torch.zeros_like(start[:, :1]), start], 1)
        return start.long().cumsum(1).clamp(max=span_max - 1), s


def test_max_len_off():
    gen = torch.Generator().manual_seed(7)
    m = build("span", span_max_len=0).eval()
    perturb(m, gen)
    x = torch.randint(0, 1000, (2, 256), generator=gen).to(DEV)
    tune_theta(m, x)
    cu = torch.tensor([0, 90, 256, 300, 512])
    with torch.no_grad():
        for tag, kw in (("cu=None", {}), ("cu", dict(cu=cu.to(DEV), cu_cpu=cu))):
            y = m(x, **kw)
            okkie_span.span_ids = old_span_ids
            try:
                y_old = m(x, **kw)
            finally:
                okkie_span.span_ids = span_ids
            check(f"span_max_len = 0 == pre-fix span ids ({tag}, fp32, bitwise)", torch.equal(y, y_old),
                  f"max diff {(y - y_old).abs().max().item():.1e}")


def test_length_one():
    gen = torch.Generator().manual_seed(8)
    m = build("span").eval()
    perturb(m, gen)
    x = torch.randint(0, 1000, (3, 1), generator=gen).to(DEV)
    cu = torch.tensor([0, 1, 40, 41, 64])  # documents of length 1 inside a flattened row
    x2 = torch.randint(0, 1000, (1, 64), generator=gen).to(DEV)
    try:
        with torch.no_grad():
            y = m(x)
            y2 = m(x2, cu=cu.to(DEV), cu_cpu=cu)
        ok = y.shape == (3, 1, 1000) and bool(torch.isfinite(y).all()) and y2.shape == (1, 64, 1000) \
            and bool(torch.isfinite(y2).all())
        info = f"shapes {tuple(y.shape)}, with length-1 documents {tuple(y2.shape)}"
    except Exception as e:  # noqa: BLE001
        ok, info = False, repr(e)
    check("L=1 forward (and length-1 documents with cu)", ok, info)


def test_forced_real(name="okkie-transformer-26m", rows=4):
    """Initial-state model of the run config, default theta, 2048-token rows of data/switch: spans per row >= 2048 /
    span_max_len and no span longer than span_max_len; with cu (unequal documents) lengths restart per document;
    backward: rho gets a nonzero gradient (o_read != 0). span_max_len = 0: rho grad exactly 0 (the old state)."""
    cfg = load_cfg(name)["model"]
    T, ML = 2048, cfg["span_max_len"]
    buf = next(switch_batches(1, rows, T, seed=3))
    x, y = torch.from_numpy(buf[:, :-1]).to(DEV), torch.from_numpy(buf[:, 1:]).to(DEV)
    for ml in (ML, 0):
        torch.manual_seed(1234)
        model = LM(ModelConfig(**dict(cfg, span_max_len=ml))).to(DEV).train()
        record(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        ns = [int(s.max(1).values.min()) + 1 for s in sids(model)]
        nmax = [int(s.max(1).values.max()) + 1 for s in sids(model)]
        longest = max(int(run_lengths(r).max()) for s in sids(model) for r in s)
        rho = [sp.rho.grad.abs().max().item() for sp in spans_of(model)]
        rho_n = [sp.rho.grad.norm().item() for sp in spans_of(model)]
        if ml:
            check(f"{name} init: spans/row >= {T // ML}, longest span <= {ML} (span_max_len {ML})",
                  min(ns) >= T // ML and longest <= ML,
                  f"spans/row per layer min {ns} max {nmax}, longest span {longest}")
            check(f"{name} init: rho grad nonzero in every layer", all(v > 0 for v in rho),
                  "|grad rho| L2 per layer " + " ".join(f"{v:.2e}" for v in rho_n))
        else:
            check(f"{name} init, span_max_len 0: 1 span/row and rho grad exactly 0 (the problem being fixed)",
                  max(nmax) == 1 and all(v == 0 for v in rho), f"spans/row max {nmax}, max|grad rho| {max(rho)}")
        # cu: unequal documents in one flattened row; forced lengths restart at each document
        if ml:
            model.eval()
            cu = torch.tensor([0, 100, 1000, 1001, 2048 + 300, 2 * T])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x[:2], None, cu.to(DEV), cu)
            lens = (cu[1:] - cu[:-1]).tolist()
            ok, longest, per_doc = True, 0, []
            for s in sids(model):
                nd = []
                for d, r in enumerate(s.cpu()):
                    r = r[:lens[d]]
                    rl = run_lengths(r)
                    longest = max(longest, int(rl.max()))
                    nd.append(len(rl))
                    ok &= int(r[0]) == 0 and len(rl) >= min(-(-lens[d] // ML), cfg["span_max"])
                per_doc.append(nd)
            check(f"{name} init with cu: forced span lengths restart per document", ok and longest <= ML,
                  f"doc lens {lens}, spans per doc (layer 0) {per_doc[0]}, longest span {longest}")
        record(model, False)
        del model, loss
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- init ~ linear
def test_init_vs_linear():
    gen = torch.Generator().manual_seed(2)
    lin, sp = build("linear", seed=5).eval(), build("span", seed=5).eval()
    sl, ss = lin.state_dict(), sp.state_dict()
    shared_eq = all(torch.equal(sl[k], ss[k]) for k in sl)
    check("shared params identical to linear (same seed)", shared_eq and set(sl) <= set(ss),
          f"{len(sl)} shared, {len(ss) - len(sl)} new")
    x = torch.randint(0, 1000, (2, 256), generator=gen).to(DEV)
    tune_theta(sp, x)  # spans present, but rho = 0 so reads do not enter
    with torch.no_grad():
        yl = lin(x)
        ys = sp(x)
        rel = ((ys - yl).norm() / yl.norm()).item()
        check("init (rho=0, b=6, tau=0) close to linear (fp32)", rel < 2e-2, f"rel L2 diff {rel:.2e}, "
              f"max abs {(ys - yl).abs().max().item():.2e}")
        for s in spans_of(sp):
            s.gate_b.fill_(100.0)  # sigmoid(100) == 1.0 exactly in fp32 -> w = 1
        ys = sp(x)
        d = (ys - yl).abs().max().item()
        check("w forced to 1 matches linear (fp32)", d < 1e-4, f"max abs diff {d:.2e}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yl, ys = lin(x).float(), sp(x).float()
        d = (ys - yl).abs().max().item()
        check("w forced to 1 matches linear (bf16 autocast)", d < 5e-2, f"max abs diff {d:.2e} "
              f"(logit scale {yl.abs().max().item():.2f})")
        cu = torch.tensor([0, 90, 256, 300, 512])
        yl, ys = lin(x, cu=cu.to(DEV), cu_cpu=cu), sp(x, cu=cu.to(DEV), cu_cpu=cu)
        d = (ys - yl).abs().max().item()
        check("w forced to 1 matches linear with cu (fp32)", d < 1e-4, f"max abs diff {d:.2e}")


# ---------------------------------------------------------------------------------------------- read path
def ref_read(q, mung, sid, sim, restore_w, coarse_w, log_scale, m, top_n):
    """Loop reference: restore only for the selected high spans of each token."""
    N, L, H, Dh = q.shape
    S = mung.shape[1]
    out = torch.zeros_like(q)
    for n in range(N):
        for t in range(L):
            usable = list(range(int(sid[n, t])))
            if not usable:
                continue
            hi = sorted(usable, key=lambda j: -sim[n, t, j].item())[:top_n]
            lo = [j for j in usable if j not in hi]
            for h in range(H):
                ks, vs = [], []
                for j in hi:
                    kv = (restore_w @ mung[n, j, h]).view(m, 2, Dh)
                    ks.append(kv[:, 0]), vs.append(kv[:, 1])
                for j in lo:
                    kv = (coarse_w @ mung[n, j, h]).view(2, Dh)
                    ks.append(kv[None, 0]), vs.append(kv[None, 1])
                K, V = torch.cat(ks), torch.cat(vs)
                sc = torch.cat([(l2n(K) @ l2n(q[n, t, h])) * log_scale[h].exp(), q.new_zeros(1)])
                p = torch.softmax(sc, 0)
                out[n, t, h] = p[:-1] @ V
    return out


def test_read_path():
    gen = torch.Generator().manual_seed(3)
    N, L, H, Dh, S, R, m, top_n = 2, 40, 2, 8, 6, 5, 3, 2
    q = torch.randn(N, L, H, Dh, generator=gen, dtype=torch.float64)
    mung = torch.randn(N, S, H, R, generator=gen, dtype=torch.float64)
    rw = torch.randn(m * 2 * Dh, R, generator=gen, dtype=torch.float64)
    cw = torch.randn(2 * Dh, R, generator=gen, dtype=torch.float64)
    ls = torch.randn(H, generator=gen, dtype=torch.float64)
    starts = torch.zeros(N, L, dtype=torch.long)
    starts[0, [5, 9, 20, 26, 33]] = 1
    starts[1, [12, 30]] = 1
    sid = starts.cumsum(1).clamp(max=S - 1)
    sim = torch.randn(N, L, S, generator=gen, dtype=torch.float64)
    high, low = read_masks(sid, sim, top_n, S)
    usable = torch.arange(S) < sid[..., None]
    ok = bool(((high | low) == usable).all() and not (high & low).any()
              and (high.sum(-1) == usable.sum(-1).clamp(max=top_n)).all())
    check("read masks: high | low = usable, disjoint, |high| = min(n, #usable)", ok)
    o = span_read(q, mung, high, low, rw, cw, ls, m)
    ref = ref_read(q, mung, sid, sim, rw, cw, ls, m, top_n)
    d = (o - ref).abs().max().item()
    check("restore-all-then-mask == restore-selected-only (float64 loop reference)", d < 1e-10, f"max diff {d:.1e}")
    z0 = (o[sid == 0] == 0).all().item()
    check("token with no finished span reads exactly 0", z0 and (sid == 0).sum() > 0,
          f"{int((sid == 0).sum())} such tokens")
    # spans that are not usable (current / later) never matter: replace their mung, output unchanged at t
    t = 22  # sample 0: sid 3 -> usable {0,1,2}
    mung2 = mung.clone()
    mung2[0, 3:] = torch.randn(S - 3, H, R, generator=gen, dtype=torch.float64)
    o2 = span_read(q, mung2, high, low, rw, cw, ls, m)
    check("current/later spans do not enter the read", torch.equal(o2[0, t], o[0, t]))
    # in the model: tokens of span 0 read exactly 0
    mdl = build("span").eval()
    perturb(mdl, gen)
    x = torch.randint(0, 1000, (2, 256), generator=gen).to(DEV)
    tune_theta(mdl, x)
    record(mdl)
    with torch.no_grad():
        mdl(x)
    ok = all(bool((s.last["o_read"][s.last["sid"] == 0] == 0).all()) for s in spans_of(mdl))
    check("model: o_read == 0 exactly for tokens in span 0", ok)
    record(mdl, False)


# ---------------------------------------------------------------------------------------------- gradients
def test_grads():
    gen = torch.Generator().manual_seed(4)
    m = build("span").train()
    perturb(m, gen)
    x = torch.randint(0, 1000, (2, 256), generator=gen).to(DEV)
    tune_theta(m, x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x, x)
    loss.backward()
    names = ["z_w1", "z_w2", "mung_w", "restore_w", "coarse_w", "read_log_scale", "tau", "gate_b", "rho"]
    for li, sp in enumerate(spans_of(m)):
        g = {n: getattr(sp, n).grad for n in names}
        bad = [n for n, v in g.items() if v is None or not torch.isfinite(v).all() or v.abs().max() == 0]
        check(f"grads reach new params (layer {li}, bf16 autocast, ckpt read)", not bad,
              "zero/None: " + str(bad) if bad else " ".join(f"{n}={v.norm().item():.1e}" for n, v in g.items()))


# ---------------------------------------------------------------------------------------------- kernels vs torch
def relerr(a, b):
    """||a - b|| / ||b|| (0 when both are 0)."""
    a, b = a.double(), b.double()
    d, n = (a - b).norm().item(), b.norm().item()
    return d / n if n > 0 else (0.0 if d == 0 else float("inf"))


def forced_sid(N, L, K, mode="equal", seed=0):
    """K spans per row. equal: boundaries at multiples of L/K; random: K-1 random distinct boundaries in [1, L)."""
    if mode == "equal":
        return (torch.arange(L, device=DEV) * K // L)[None].expand(N, L).contiguous()
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(N):
        st = torch.zeros(L, dtype=torch.long)
        st[torch.randperm(L - 1, generator=g)[:K - 1] + 1] = 1
        rows.append(st.cumsum(0))
    return torch.stack(rows).to(DEV)


def out_grads(fn, inputs, gout):
    xs = [x.detach().clone().requires_grad_() for x in inputs]
    out = fn(*xs)
    out.backward(gout)
    return out.detach(), [x.grad for x in xs]


def test_kernels():
    """Triton lifetime gate / span read vs the torch references, float32 inputs: out rel L2 < 1e-5, grads < 2e-4.
    Read: K spans per row (equal 1, 2, 8, 32; random 32; random 64 clamped to span_max 32), lowp 0 (fp32 dots) and
    2 (3x bf16 split dots, used in training)."""
    gen = torch.Generator().manual_seed(9)
    rn = lambda *s: torch.randn(*s, generator=gen).to(DEV)  # noqa: E731
    N, L, H, P, Dh, Sx, R, m = 2, 1024, 3, 32, 64, 32, 64, 4
    gw = rn(N, L, H)
    c, g, beta = rn(N, L, H, P), -torch.rand(N, L, H, generator=gen).to(DEV) * 0.05, torch.rand(N, L, H, generator=gen).to(DEV)
    for tag, tau, bias in (("perturbed", rn(H) * 2, rn(H)), ("init", torch.zeros(H, device=DEV),
                                                             torch.full((H,), 6.0, device=DEV))):
        ins = [c, g, beta, tau, bias]
        oa, ga = out_grads(lambda *a: okkie_span.lifetime_gate(*a, 64), ins, gw)
        ob, gb = out_grads(lambda *a: okkie_span.lifetime_gate_triton(*a, 64), ins, gw)
        eo, eg = relerr(ob, oa), [relerr(b, a) for a, b in zip(ga, gb)]
        check(f"kernel lifetime gate ({tag}) == torch", eo < 1e-5 and max(eg) < 2e-4,
              f"out rel {eo:.1e}, grads (c, g, beta, tau, b) rel " + " ".join(f"{e:.1e}" for e in eg))
    q, mung, go = rn(N, L, H, Dh), rn(N, Sx, H, R), rn(N, L, H, Dh)
    rw, cw = rn(m * 2 * Dh, R) * R ** -0.5, rn(2 * Dh, R) * R ** -0.5
    ls = torch.full((H,), 0.5 * math.log(Dh), device=DEV) + rn(H) * 0.3
    for K in (1, 2, 8, 32, "r32", "r64"):
        sid = forced_sid(N, L, K) if isinstance(K, int) else forced_sid(N, L, int(K[1:]), "random").clamp(max=Sx - 1)
        sim = rn(N, L, Sx)
        hi, lo = read_masks(sid, sim, 2, Sx)
        top = okkie_span.top_idx(sid, sim, 2, Sx)
        ins = [q, mung, rw, cw, ls]
        oa, ga = out_grads(lambda q_, mu, r_, c_, l_: span_read(q_, mu, hi, lo, r_, c_, l_, m), ins, go)
        for lowp in (0, 2):
            ob, gb = out_grads(lambda q_, mu, r_, c_, l_: okkie_span.span_read_triton(q_, mu, sid, top, r_, c_, l_, m,
                                                                                         lowp), ins, go)
            eo, eg = relerr(ob, oa), [relerr(b, a) for a, b in zip(ga, gb)]
            ok = eo < 1e-5 and max(eg) < 2e-4 and (K != 1 or (ob == 0).all().item())
            check(f"kernel span read K={K} lowp={lowp} == torch", ok,
                  f"spans/row {int(sid.max()) + 1}, out rel {eo:.1e}, grads (q, mung, restore, coarse, log_scale) "
                  f"rel " + " ".join(f"{e:.1e}" for e in eg))


def ref_forward(self, p, x, q, k, v, cu=None, cu_cpu=None):
    """OkkieSpan.forward before the kernels (torch: L x L EMA, lifetime_gate, one-hot means over span_max,
    read_masks + span_read with checkpoint). Reference for test_layer_vs_torch."""
    B, T = x.shape[:2]
    H, Dh = self.H, self.Dh
    lay = okkie_span.DocLayout(B, T, cu, cu_cpu, x.device)
    z = self.features(x)
    a, b = p.ab(x).chunk(2, -1)
    with torch.autocast("cuda", enabled=False):
        zf = z.float()
        g = -torch.exp(p.A_log.float()) * F.softplus(a.float() + p.dt_bias.float())
        beta = torch.sigmoid(b.float())
        zd = lay.to_doc(zf)
        N, L = zd.shape[:2]
        with torch.no_grad():
            zv = zd * lay.valid[..., None] if lay.valid is not None else zd
            s = F.cosine_similarity(ema(zv, self.h_short), ema(zv, self.h_long), dim=-1, eps=1e-8)
            start = (s[:, 1:] < self.theta) & (s[:, :-1] >= self.theta)
            start = torch.cat([start.new_zeros(start.shape[0], 1), start], 1)
            sid = starts_to_ids(start, self.max_len).clamp(max=self.S - 1)
        w = lay.from_doc(okkie_span.lifetime_gate(zd.view(N, L, H, self.P), lay.to_doc(g), lay.to_doc(beta),
                                                  self.tau.float(), self.gate_b.float(), self.chunk))
        oh = F.one_hot(sid, self.S).float()
        if lay.valid is not None:
            oh = oh * lay.valid[..., None]
        oh = oh / oh.sum(1, keepdim=True).clamp(min=1)
        kv = lay.to_doc(torch.cat([k, v], -1).float())
        mung = torch.einsum("nshd,hrd->nshr", torch.einsum("nls,nlhd->nshd", oh, kv),
                            self.mung_w.float().view(H, self.R, 2 * Dh))
        with torch.no_grad():
            zbar = torch.einsum("nls,nld->nsd", oh, zd)
            sim = torch.einsum("nld,nsd->nls", F.normalize(zd, dim=-1, eps=1e-8), F.normalize(zbar, dim=-1, eps=1e-8))
            high, low = read_masks(sid, sim, self.top_n, self.S)
        args = (lay.to_doc(q.float()), mung, high, low, self.restore_w.float(), self.coarse_w.float(),
                self.read_log_scale.float(), self.m)
        o_read = lay.from_doc(torch.utils.checkpoint.checkpoint(span_read, *args, use_reentrant=False))
    if self.record:
        self.last = {"sid": sid.detach(), "valid": lay.valid}
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    o, _ = chunk_gated_delta_rule(q, k, v, g=w * g, beta=w * beta, use_qk_l2norm_in_kernel=True,
                                  use_beta_sigmoid_in_kernel=False, use_gate_in_kernel=False, cu_seqlens=cu,
                                  cu_seqlens_cpu=cu_cpu)
    o = (o.float() * w[..., None] + self.rho.float()[:, None] * o_read).to(o.dtype)
    o = p.o_norm(o, p.g_proj(x).unflatten(-1, (H, Dh)))
    return p.o_proj(o.flatten(-2))


def test_layer_vs_torch():
    """One span mixer, new forward (chunk EMA, Triton gate / read, span count from the batch, bf16 span sums) vs
    ref_forward, out and grads (x and all mixer params). Theta tuned so several spans occur, rho != 0.
    fp32: same span ids, out rel < 1e-4, grads rel < 1e-3. bf16 autocast: rel diff <= 1.5 x the noise floor (ref
    vs ref with x shifted by 1e-6 noise, max of 2 draws)."""
    gen = torch.Generator().manual_seed(10)
    mdl = build("span", span_max=16, span_max_len=64).train()
    perturb(mdl, gen)
    T = 512
    tok = torch.randint(0, 1000, (2, T), generator=gen).to(DEV)
    cu = torch.tensor([0, 90, 256, 300, 700, 1024])
    for tag, kw, shape in (("cu=None", {}, (2, T)), ("cu (5 docs)", dict(cu=cu.to(DEV), cu_cpu=cu), (1, 2 * T))):
        tune_theta(mdl, tok.view(shape), **kw)
        mix = mdl.blocks[0].mixer
        sp = mix.span
        x0 = torch.randn(*shape, 128, generator=gen).to(DEV)
        gout = torch.randn(*shape, 128, generator=gen).to(DEV)
        params = [(n, p_) for n, p_ in mix.named_parameters()]

        def run(fwd, x, bf16):
            mix.zero_grad(set_to_none=True)
            x = x.detach().clone().requires_grad_()
            orig = okkie_span.OkkieSpan.forward
            okkie_span.OkkieSpan.forward = fwd
            sp.record = True
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    y = mix(x, **kw)
                y.float().backward(gout)
            finally:
                okkie_span.OkkieSpan.forward = orig
                sp.record = False
            sid = sp.last["sid"].clone()
            gr = [x.grad.clone()] + [torch.zeros_like(p_) if p_.grad is None else p_.grad.clone() for _, p_ in params]
            return y.detach().float(), gr, sid

        new = okkie_span.OkkieSpan.forward
        for bf16 in (False, True):
            ya, ga, sa = run(ref_forward, x0, bf16)
            yb, gb, sb = run(new, x0, bf16)
            eo, eg = relerr(yb, ya), [relerr(b, a) for a, b in zip(ga, gb)]
            names = ["x"] + [n for n, _ in params]
            worst = max(range(len(eg)), key=lambda i: eg[i])
            ns = int(sa.max()) + 1
            if not bf16:
                ok = torch.equal(sa, sb) and eo < 1e-4 and max(eg) < 1e-3
                check(f"layer new == torch reference ({tag}, fp32)", ok,
                      f"spans {ns}, same span ids {torch.equal(sa, sb)}, out rel {eo:.1e}, grads max rel "
                      f"{eg[worst]:.1e} ({names[worst]}), rho grad rel {eg[names.index('span.rho')]:.1e}")
            else:
                fo, fg = 0.0, [0.0] * len(eg)
                for s in (1, 2):
                    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(s)).to(DEV) * 1e-6
                    yc, gc, _ = run(ref_forward, x0 + noise, bf16)
                    fo = max(fo, relerr(yc, ya))
                    fg = [max(f, relerr(c_, a)) for f, c_, a in zip(fg, gc, ga)]
                bad = [names[i] for i in range(len(eg)) if eg[i] > 1.5 * fg[i] and eg[i] > 1e-6]
                ok = eo <= 1.5 * fo and not bad
                check(f"layer new ~ torch reference ({tag}, bf16 autocast, within noise floor)", ok,
                      f"spans {ns}, same span ids {torch.equal(sa, sb)}, out rel {eo:.1e} (floor {fo:.1e}), grads "
                      f"worst rel {eg[worst]:.1e} ({names[worst]}, floor {fg[worst]:.1e})"
                      + (f", above 1.5 x floor: {bad}" if bad else ""))


# ---------------------------------------------------------------------------------------------- linear / pool
def ref_logits():
    """Same input as the pre-edit reference script (2 x 256, fixed tokens; cu=None and cu)."""
    out = {}
    for run in ["plain_20261110_231750", "okkie-transformer-117M_20261110_231751"]:
        d = ROOT / "model" / run
        cfg = tomllib.load(open(d / "config.toml", "rb"))
        m = LM(ModelConfig(**cfg["model"])).to(DEV).eval()
        m.load_state_dict(torch.load(d / "step_007629.pt", map_location=DEV, weights_only=False)["model"])
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 32768, (2, 256), generator=g).to(DEV)
        cu = torch.tensor([0, 100, 256, 300, 512])
        with torch.no_grad():
            for tag, kw in [("nocu", {}), ("cu", dict(cu=cu.to(DEV), cu_cpu=cu))]:
                out[f"{run}/{tag}/fp32"] = m(x, **kw).float().cpu()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out[f"{run}/{tag}/bf16"] = m(x, **kw).float().cpu()
        del m
        torch.cuda.empty_cache()
    return out


def test_unchanged(ref_path):
    ref = torch.load(ref_path)
    out = ref_logits()
    for k in ref:
        check(f"unchanged vs pre-edit: {k}", torch.equal(ref[k], out[k]),
              "" if torch.equal(ref[k], out[k]) else f"max diff {(ref[k] - out[k]).abs().max().item():.2e}")


# ---------------------------------------------------------------------------------------------- speed / spans
def load_cfg(name):
    return tomllib.load(open(ROOT / "configs" / f"{name}.toml", "rb"))


def switch_batches(n_batches, B, T, seed=0):
    m = np.memmap(ROOT / "data/switch/switch/train.bin", dtype=np.uint16, mode="r")
    rows = (len(m) - 1) // T
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        idx = rng.choice(rows, B, replace=False)
        yield np.stack([m[i * T:i * T + T + 1] for i in idx]).astype(np.int64)


def tensors(buf):
    """doc_mask = false (all four run configs): no cu."""
    x = torch.from_numpy(buf[:, :-1]).to(DEV)
    y = torch.from_numpy(buf[:, 1:]).to(DEV)
    return x, y, None, None


RUNS = ["plain-26m", "okkie-transformer-26m", "plain-60m", "okkie-transformer-60m"]


def test_speed(steps=8):
    for name in RUNS:
        cfg = load_cfg(name)
        B = cfg["train"]["micro_batch"]
        torch.manual_seed(cfg["data"]["seed"])
        model = LM(ModelConfig(**cfg["model"])).to(DEV).train()
        n = model.num_params()
        bufs = list(switch_batches(steps, B, cfg["data"]["seq_len"]))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i, buf in enumerate(bufs):
            x, y, cu, cu_cpu = tensors(buf)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y, cu, cu_cpu)
            loss.backward()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            model.zero_grad(set_to_none=True)
        dt = sum(times[3:]) / len(times[3:])
        print(f"[speed] {name}: params {n} ({n / 1e6:.3f}M), fwd+bwd B={B} T={x.shape[-1]} "
              f"{B * x.shape[-1] / dt:.0f} tok/s ({dt * 1e3:.0f} ms/micro-step, no torch.compile), "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
        if cfg["model"]["gate_mode"] == "span":
            model.eval()
            record(model)
            per_layer = []
            with torch.no_grad():
                for buf in switch_batches(4, 8, cfg["data"]["seq_len"], seed=1):
                    x, y, cu, cu_cpu = tensors(buf)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model(x, None, cu, cu_cpu)
                    per_layer.append([(s.last["sid"].max(1).values + 1).float() for s in spans_of(model)])
            for li in range(len(per_layer[0])):
                v = torch.cat([p[li] for p in per_layer])
                print(f"[spans] layer {li}: spans/seq at default theta={spans_of(model)[li].theta}: mean "
                      f"{v.mean().item():.2f} min {v.min().item():.0f} max {v.max().item():.0f} "
                      f"(32 rows of data/switch, random init)", flush=True)
            record(model, False)
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- train.py smoke
def test_smoke(out, steps=6):
    """train.py on each run config with max_steps = steps (eval + save at the last step), in its own process on
    DEV (HIP_VISIBLE_DEVICES, as Train/run1.sh selects a GPU). Checks: exit 0, finite losses, last < first, eval
    record, step checkpoint loads into a fresh model with matching keys."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = str(torch.device(DEV).index or 0)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=gpu, HIPBLASLT_TENSILE_LIBPATH=str(ROOT / "hipblaslt-lib/gfx1201"))
    for name in RUNS:
        cfg = (ROOT / "configs" / f"{name}.toml").read_text()
        cfg = cfg.replace("max_steps = 0", f"max_steps = {steps}").replace("log_every = 5", "log_every = 1")
        cfg = cfg.replace("eval_every = 250", f"eval_every = {steps}").replace("eval_batches = 16", "eval_batches = 2")
        cfg = cfg.replace("save_every = 250", f"save_every = {steps}")
        cpath = out / f"{name}.toml"
        cpath.write_text(cfg)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / "train.py"), "--config", str(cpath), "--out", str(out)],
                           env=env, cwd=ROOT, capture_output=True, text=True)
        (out / f"{name}.log").write_text(r.stdout + r.stderr)
        runs = sorted(out.glob(f"{name}_2*"))
        if r.returncode != 0 or not runs:
            check(f"smoke train.py {name}", False, f"rc {r.returncode}: {(r.stdout + r.stderr)[-2000:]}")
            continue
        recs = [json.loads(l) for l in open(runs[-1] / "log.jsonl")]
        losses = [x["loss"] for x in recs if "loss" in x]
        evals = [x for x in recs if "val" in x]
        tps = [x["tok_s"] for x in recs if "tok_s" in x]
        ck = torch.load(runs[-1] / f"step_{steps:06d}.pt", map_location="cpu", weights_only=False)
        c = tomllib.loads(cfg)
        with torch.device("meta"):
            fresh = LM(ModelConfig(**c["model"]))
        keys_ok = set(ck["model"]) == set(fresh.state_dict()) and (runs[-1] / "latest.pt").exists()
        ok = (all(np.isfinite(losses)) and len(losses) == steps and losses[-1] < losses[0] and len(evals) == 1
              and np.isfinite(evals[0]["val_mean"]) and keys_ok)
        check(f"smoke train.py {name}", ok, f"loss {losses[0]:.3f} -> {losses[-1]:.3f}, val {evals[0]['val_mean']:.3f}"
              f" ckpt keys ok {keys_ok}, tok/s (last 4 steps, compiled, incl. data) {np.mean(tps[-4:]):.0f}, "
              f"mem {recs[-2].get('mem_gb', 0):.1f} GB, wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--skip-unit", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    DEV = a.device
    torch.cuda.set_device(DEV)
    if not a.skip_unit:
        test_kernels()
        test_layer_vs_torch()
        test_read_path()
        test_causality()
        test_causality(tune=False, tag=" (default theta, span_max_len 16)", span_max_len=16, span_max=32)
        test_init_vs_linear()
        test_grads()
        test_forced_ids()
        test_max_len_off()
        test_length_one()
        test_forced_real()
    if a.ref:
        test_unchanged(a.ref)
    if a.speed:
        test_speed()
    if a.smoke:
        test_smoke(a.smoke)
    print(f"{sum(ok for _, ok in RESULTS)}/{len(RESULTS)} passed", flush=True)
    if not all(ok for _, ok in RESULTS):
        sys.exit(1)
