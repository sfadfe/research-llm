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
from okkie_span import read_masks, span_ids, span_read, l2n  # noqa: E402

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
def test_causality():
    gen = torch.Generator().manual_seed(1)
    m = build("span").eval()
    perturb(m, gen)
    T = 256
    x = torch.randint(0, 1000, (2, T), generator=gen).to(DEV)
    tune_theta(m, x)
    record(m)
    with torch.no_grad():
        y0 = m(x)
        ns = n_spans(m)
        nread = [(sp.last["high"].any(-1)).float().mean().item() for sp in spans_of(m)]
        for t in (37, 130, 200):
            x1 = x.clone()
            x1[:, t] = (x1[:, t] + 1 + torch.randint(0, 998, (2,), generator=gen).to(DEV)) % 1000
            y1 = m(x1)
            d_before = (y1[:, :t] - y0[:, :t]).abs().max().item()
            d_after = (y1[:, t:] - y0[:, t:]).abs().max().item()
            check(f"causality cu=None t={t}", d_before < 1e-4 and d_after > 1e-3,
                  f"max|d| before t {d_before:.2e}, at/after t {d_after:.2e}; spans/seq per layer {ns}, "
                  f"frac tokens with a high span {[round(v, 3) for v in nread]}")
    # flattened with cu: unequal documents (gather layout); changes stay inside the document and after t
    cu = torch.tensor([0, 90, 256, 300, 512])
    kw = dict(cu=cu.to(DEV), cu_cpu=cu)
    tune_theta(m, x, **kw)
    with torch.no_grad():
        y0 = m(x, **kw)[0]
        ns = n_spans(m)
        for t, end in ((40, 90), (150, 256), (270, 300)):
            x1 = x.clone().view(-1)
            x1[t] = (x1[t] + 7) % 1000
            y1 = m(x1.view(2, T), **kw)[0]
            d_before = (y1[:t] - y0[:t]).abs().max().item()
            d_other = (y1[end:] - y0[end:]).abs().max().item()
            d_in = (y1[t:end] - y0[t:end]).abs().max().item()
            check(f"causality+doc isolation cu t={t}", d_before < 1e-4 and d_other < 1e-4 and d_in > 1e-3,
                  f"before {d_before:.2e}, later docs {d_other:.2e}, own doc after t {d_in:.2e}; spans/doc {ns}")
    record(m, False)


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
        test_read_path()
        test_causality()
        test_init_vs_linear()
        test_grads()
    if a.ref:
        test_unchanged(a.ref)
    if a.speed:
        test_speed()
    if a.smoke:
        test_smoke(a.smoke)
    print(f"{sum(ok for _, ok in RESULTS)}/{len(RESULTS)} passed", flush=True)
    if not all(ok for _, ok in RESULTS):
        sys.exit(1)
