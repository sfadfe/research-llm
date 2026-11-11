"""Post-training evaluation of one run.

Usage (from the research-llm directory):
    .venv/bin/python Eval/evaluate.py --run model/<run dir> --device cuda:<N> [--ckpt step_XXXXXX.pt|latest.pt|<path>]

Writes results/<run name>/{ppl,position_loss,gates,train}.json, pool.json (pooled arms only) and summary.json.
summary.json is removed at start and written last, so its presence marks a complete evaluation. Re-running overwrites.

Validation data: data/bin/<source>/val.bin, cut into consecutive rows of seq_len+1 tokens at offsets i*seq_len exactly
like the old Train/ppl.py (row count truncated to a multiple of 8, ~1M tokens per source), batches of 8 rows, document
masking as in training (Train/data.py tensors: cu_seqlens = row starts U (positions after each eot), rows flattened).
"""
import argparse
import json
import math
import os
import sys
import time
import tomllib
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_hip = ROOT / "hipblaslt-lib" / "gfx1201"
if _hip.is_dir():  # same hipBLASLt kernels as training (Train/run1.sh); must be set before torch loads
    os.environ.setdefault("HIPBLASLT_TENSILE_LIBPATH", str(_hip))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(ROOT / "Train"))
import model as M  # noqa: E402

MB = 8                      # rows per batch (old ppl.py)
QS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
NBINS = 100                 # histogram bins on [0, 1] for alpha and beta
L1MA = (-8.0, 0.0, 80)      # extra histogram of log10(1 - alpha): alpha piles up near 1, where [0,1] bins are too coarse
SAMPLE_STRIDE = 4           # every 4th token is kept for exact quantiles (mean/std/hist use every token)
WIN_ROWS = 8                # rows per source (row 0 of the first 8 batches) for the realised pooling-window statistics
WIN_MIN_CTX = 512           # second window summary restricted to positions with >= this many tokens of document context
WIN_ELEMS = 2 ** 23         # window stats: query rows per pass = WIN_ELEMS // T (one pass, unchanged, for T <= 2896)
LOGIT_CHUNK = 4096          # tokens per chunk for the per-token loss (LM head)
ARMS = {"linear": "A", "pool": "B", "pool_g0": "C", "pool_s0": "D"}


def die(msg):
    print(f"[evaluate] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ---------------------------------------------------------------- alpha / beta exactly as the fla kernel computes them
# Installed fla 0.6.0 (.venv/lib/python3.12/site-packages/fla). Train/model.py calls chunk_gated_delta_rule(q, k, v,
# g=a, beta=b, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, use_gate_in_kernel=True, A_log, dt_bias).
#  beta: ops/gated_delta_rule/chunk.py:286-288  beta = fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0)
#        ops/common/gate.py:29-31 (kernel)      y = scale * sigmoid(float32(x)), stored float32 (gate.py:59)
#        -> beta_t = sigmoid(float32(b_t))      (allow_neg_eigval defaults to False)
#  alpha: ops/gated_delta_rule/chunk.py:51-60   g = gdn_gate_chunk_cumsum(g, A_log, dt_bias, scale=RCP_LN2, ...)
#        ops/gated_delta_rule/gate.py:94-98     b_g = float32(g) + float32(dt_bias); gate = -exp(float32(A_log)) * softplus(b_g)
#        then gate.py:100,104-105               chunk-local cumsum, times 1/ln2 (log2 domain for the exp2-based kernels)
#        ops/utils/softplus.py:86-87,112-114    non-NVIDIA (ROCm): softplus(x) = where(x < 20, log(1 + exp(x)), x)
#        ops/utils/op.py:44-45                  exp(x) = tl.exp(float32(x)) (FLA_USE_FAST_OPS unset)
#        -> log alpha_t = -exp(A_log) * softplus(a_t + dt_bias),  alpha_t = exp(log alpha_t)  (per-token state decay)
#  (same formula: ops/gated_delta_rule/gate.py:21-46 naive_gdn_gate; Eval/test_evaluate.py checks it numerically.)
def kernel_softplus(x):
    return torch.where(x < 20.0, torch.log1p(torch.exp(x)), x)


def log_alpha(a, A_log, dt_bias):
    return -torch.exp(A_log.float()) * kernel_softplus(a.float() + dt_bias.float())


def beta_of(b):
    return torch.sigmoid(b.float())


class KernelGuard:
    """Wraps Train/model.py's chunk_gated_delta_rule for one forward to check the assumptions behind log_alpha/beta_of:
    the flags passed, and that last_gate a/b are the tensors the kernel receives."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self.orig = M.chunk_gated_delta_rule

        def wrap(*args, **kw):
            self.calls.append(kw)
            return self.orig(*args, **kw)

        M.chunk_gated_delta_rule = wrap
        return self

    def __exit__(self, *exc):
        M.chunk_gated_delta_rule = self.orig

    def check(self, gdn):
        if len(self.calls) != len(gdn):
            raise RuntimeError(f"expected {len(gdn)} chunk_gated_delta_rule calls, saw {len(self.calls)}")
        for li, (kw, m) in enumerate(zip(self.calls, gdn)):
            want = {"use_gate_in_kernel": True, "use_beta_sigmoid_in_kernel": True}
            got = {k: kw.get(k, False) for k in want}
            if got != want or kw.get("allow_neg_eigval", False):
                raise RuntimeError(f"layer {li}: kernel flags {got}, allow_neg_eigval={kw.get('allow_neg_eigval')}; "
                                   "Eval/evaluate.py alpha/beta formulas assume use_gate_in_kernel and use_beta_sigmoid_in_kernel")
            if kw.get("A_log") is not m.A_log or kw.get("dt_bias") is not m.dt_bias:
                raise RuntimeError(f"layer {li}: kernel A_log/dt_bias are not the module's parameters")
            lg = m._eval_guard
            if not (torch.equal(kw["g"], lg["a"]) and torch.equal(kw["beta"], lg["b"])):
                raise RuntimeError(f"layer {li}: last_gate a/b differ from the kernel's g/beta inputs")


# ---------------------------------------------------------------- statistics helpers
class Moments:
    """Per-head running sums (float64) + fixed-bin histogram + strided sample for quantiles, on GPU."""

    def __init__(self, H, lo, hi, bins, device, sources):
        self.H, self.lo, self.hi, self.bins = H, lo, hi, bins
        self.n = torch.zeros(H, dtype=torch.float64, device=device)
        self.s1 = torch.zeros(H, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(H, dtype=torch.float64, device=device)
        self.hist = torch.zeros(H * bins, dtype=torch.int64, device=device)
        self.src = {s: torch.zeros(2, H, dtype=torch.float64, device=device) for s in sources}
        self.sample = []

    def add(self, x, source, keep=None):
        """x (N, H) float32. keep: tensor to store for quantiles (default x) - a monotone transform of x may be passed."""
        H = self.H
        xd = x.double()
        self.n += x.shape[0]
        self.s1 += xd.sum(0)
        self.s2 += (xd * xd).sum(0)
        self.src[source][0] += x.shape[0]
        self.src[source][1] += xd.sum(0)
        idx = ((x - self.lo) / (self.hi - self.lo) * self.bins).floor().clamp(0, self.bins - 1).long()
        idx = idx + torch.arange(H, device=x.device) * self.bins
        self.hist += torch.bincount(idx.flatten(), minlength=H * self.bins)
        self.sample.append((x if keep is None else keep)[::SAMPLE_STRIDE].float().cpu())

    def head(self, h, qmap=None):
        n = float(self.n[h])
        mean = float(self.s1[h]) / n
        var = max(float(self.s2[h]) / n - mean * mean, 0.0)
        smp = torch.cat([t[:, h] for t in self.sample]).double()
        q = torch.quantile(smp, torch.tensor(QS, dtype=torch.float64)) if smp.numel() <= 16_000_000 else None
        if q is None:
            q = torch.from_numpy(np.quantile(smp.numpy(), QS))
        if qmap is not None:
            q = qmap(q)
        return {"n": int(n), "mean": mean, "std": math.sqrt(var),
                "quantiles": {str(k): float(v) for k, v in zip(QS, q)},
                "quantile_sample_n": int(smp.numel()),
                "hist": {"lo": self.lo, "hi": self.hi, "bins": self.bins,
                         "counts": self.hist[h * self.bins:(h + 1) * self.bins].tolist()},
                "per_source_mean": {s: float(v[1, h] / v[0, h]) for s, v in self.src.items() if v[0, h] > 0}}


def plain_stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "quantiles": {str(k): float(v) for k, v in zip(QS, np.quantile(x, QS))}}


# ---------------------------------------------------------------- data
def val_rows(name, T):
    m = np.memmap(ROOT / "data/bin" / name / "val.bin", dtype=np.uint16, mode="r")
    k = (len(m) - 1) // T
    k -= k % MB
    return m, k


def tensors(buf, eot, device):
    """Identical to old Train/ppl.py tensors and Train/data.py MixtureData.tensors(doc_mask=True)."""
    x, y = buf[:, :-1], buf[:, 1:]
    B, L = x.shape
    pos = np.flatnonzero(x.reshape(-1) == eot) + 1
    cu_cpu = torch.from_numpy(np.union1d(np.arange(0, B * L + 1, L), pos))
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return to(x), to(y), cu_cpu.to(device), cu_cpu


def positions(cu_cpu, n, T, row0, mm, eot):
    """For each flattened input token: position since the start of its doc-mask segment, and whether that segment
    starts at a true document start. A segment starts after an eot (always a true doc start) or at a row start; a row
    start at global offset o is a true doc start iff o == 0 or val.bin[o-1] == eot (val.bin = whole docs + eot each)."""
    cu = cu_cpu.numpy()
    seg = np.searchsorted(cu, np.arange(n), side="right") - 1
    st = cu[seg]
    p = np.arange(n) - st
    at_row = st % T == 0
    o = (row0 + st // T) * T
    true_row = (o == 0) | (mm[np.maximum(o - 1, 0)] == eot)
    return p, ~at_row | true_row


# ---------------------------------------------------------------- pooling window realised on data
def window_weights(s, gamma, seg, H, t0=0, t1=None):
    """s (T,H) float32 bounded scores (zeros for pool_s0), gamma (H,), seg (T,) segment ids of one row.
    w_{t,i} = softmax_i(s_i - gamma (t - i)) over i <= t in the same segment (Train/model.py decay_pool definition).
    Returns w (H,T,T) and dist (T,T); with t0/t1 only query rows t in [t0, t1): w (H,t1-t0,T), dist (t1-t0,T)."""
    T = s.shape[0]
    t = torch.arange(T, device=s.device)
    t1 = T if t1 is None else t1
    dist = (t[t0:t1, None] - t[None, :]).float()
    mask = (dist >= 0) & (seg[t0:t1, None] == seg[None, :])
    lg = (s.t()[:, None, :] - gamma.view(H, 1, 1) * dist).masked_fill(~mask, float("-inf"))
    return torch.softmax(lg, -1), dist


def window_stats(s, gamma, seg, H):
    """Per head: ESS = 1/sum w^2, exp(entropy), mean lookback sum w (t-i); and available context t - start + 1."""
    T = s.shape[0]
    t = torch.arange(T, device=s.device)
    qc = min(T, max(1, WIN_ELEMS // T))  # query rows per pass: bounds the (H, qc, T) tensors for long rows
    parts = []
    for t0 in range(0, T, qc):
        p, dist = window_weights(s, gamma, seg, H, t0, min(t0 + qc, T))
        parts.append((1.0 / (p * p).sum(-1), -(p * torch.log(p.clamp_min(1e-30))).sum(-1),
                      (p * dist.clamp_min(0)).sum(-1)))
        del p, dist
    ess, ent, look = (torch.cat(v, -1) for v in zip(*parts))
    newseg = torch.ones(T, dtype=torch.bool, device=s.device)
    newseg[1:] = seg[1:] != seg[:-1]
    first = torch.cummax(torch.where(newseg, t, torch.zeros_like(t)), 0).values
    ctx = (t - first + 1).float()
    return ess.cpu().numpy(), torch.exp(ent).cpu().numpy(), look.cpu().numpy(), ctx.cpu().numpy()


# ---------------------------------------------------------------- log.jsonl
def train_log(run, n_params, cfg):
    p = run / "log.jsonl"
    out = {"log": str(p)}
    if not p.exists():
        out["missing"] = True
        return out
    recs = []
    for ln in open(p):
        ln = ln.strip()
        if ln:
            try:
                recs.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    steps = [r for r in recs if "tok_s" in r]
    # first record of every (re)start includes compile/warmup: a restart shows as elapsed_h or step going backwards
    steady, prev = [], None
    for r in steps:
        if prev is not None and r["elapsed_h"] >= prev["elapsed_h"] and r["step"] > prev["step"]:
            steady.append(r["tok_s"])
        prev = r
    restarts = sum(1 for a, b in zip(steps, steps[1:]) if not (b["elapsed_h"] >= a["elapsed_h"] and b["step"] > a["step"]))
    val = {}
    for r in recs:
        if "val" in r:
            val[r["step"]] = {"step": r["step"], "val": r["val"], "val_mean": r["val_mean"]}
    out.update({
        "tok_s_mean": float(np.mean(steady)) if steady else None,
        "tok_s_median": float(np.median(steady)) if steady else None,
        "tok_s_note": "mean/median of log records' tok_s, excluding the first record after start and after each restart "
                      "(compile/warmup); eval time is not included in tok_s (train.py resets its timer after eval)",
        "tok_s_records": len(steady),
        "restarts": restarts,
        "mem_gb_max": max((r["mem_gb"] for r in steps), default=None),
        "tflops_gpu_mean_steady": None,
        "final_step": steps[-1]["step"] if steps else None,
        "final_train_loss": steps[-1]["loss"] if steps else None,
        "final_train_loss_mean_last10": float(np.mean([r["loss"] for r in steps[-10:]])) if steps else None,
        "elapsed_h_last_segment": steps[-1]["elapsed_h"] if steps else None,
        "val_curve": [val[k] for k in sorted(val)],
    })
    if steady:
        out["tflops_gpu_mean_steady"] = 6 * n_params * out["tok_s_mean"] / 1e12
    return out


# ---------------------------------------------------------------- output
def finite(o, path, bad):
    if isinstance(o, float):
        if math.isfinite(o):
            return o
        bad.append(path)
        return None
    if isinstance(o, dict):
        return {k: finite(v, f"{path}.{k}", bad) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v, f"{path}[{i}]", bad) for i, v in enumerate(o)]
    if isinstance(o, (np.floating, np.integer)):
        return finite(o.item(), path, bad)
    return o


def dump(out, name, obj, warnings):
    bad = []
    obj = finite(obj, name, bad)
    if bad:
        warnings.append(f"{name}: {len(bad)} non-finite values written as null (first: {bad[0]})")
    tmp = out / (name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
    os.replace(tmp, out / name)


def pick_ckpt(run, arg):
    if arg:
        p = Path(arg)
        p = p if p.is_absolute() or p.exists() else run / arg
        if not p.is_file():
            die(f"checkpoint {p} not found")
        return p
    steps = sorted(run.glob("step_*.pt"))
    if steps:
        return max(steps, key=lambda q: int(q.stem.split("_")[1]))
    if (run / "latest.pt").is_file():
        return run / "latest.pt"
    die(f"no step_*.pt or latest.pt in {run}")


def param_counts(model, gdn):
    tot = model.num_params()
    gate = {"ab": 0, "ab_c": 0, "c_proj": 0, "c_norm": 0, "score_w": 0, "gamma_raw": 0, "A_log": 0, "dt_bias": 0}
    for m in gdn:
        for k in gate:
            v = getattr(m, k, None)
            if isinstance(v, torch.nn.Module):
                gate[k] += sum(p.numel() for p in v.parameters())
            elif isinstance(v, torch.Tensor):
                gate[k] += v.numel()
    return {"total": tot, "embedding_tied": model.embed.weight.numel(), "non_embedding": tot - model.embed.weight.numel(),
            "gate_params_by_name": gate, "pool_extra": gate["ab_c"] + gate["c_proj"] + gate["c_norm"] + gate["score_w"] + gate["gamma_raw"]}


# ---------------------------------------------------------------- main
@torch.no_grad()
def evaluate(args):
    t0 = time.time()
    run = Path(args.run).resolve()
    if not (run / "config.toml").is_file():
        die(f"{run}/config.toml not found")
    out = Path(args.out).resolve() / run.name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        (out / "summary.json").unlink()  # stale completion marker of a previous evaluation; rewritten last
    device = torch.device(args.device)
    if device.type != "cuda":
        die("--device must be cuda:N (fla Triton kernels)")
    if device.index is None:
        device = torch.device("cuda", 0)
    if device.index >= torch.cuda.device_count():
        die(f"{device} not available (device_count={torch.cuda.device_count()}, "
            f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')})")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    cfg = tomllib.load(open(run / "config.toml", "rb"))
    meta = json.load(open(ROOT / "data/bin/meta.json"))
    eot = int(meta["eot"])
    T = int(args.seq_len or cfg["data"]["seq_len"])
    if not cfg["data"].get("doc_mask", True):
        die("config has doc_mask = false; this evaluation assumes document masking as in training")
    sources = list(cfg["data"]["mix"])
    mc = M.ModelConfig(**cfg["model"])
    ckpt = pick_ckpt(run, args.ckpt)
    print(f"[evaluate] run {run.name}  ckpt {ckpt.name}  device {device}", flush=True)
    model = M.LM(mc).to(device).eval()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    step = ck.get("step")
    del ck
    gdn = [blk.mixer for blk in model.blocks if blk.kind == "G"]
    gdn_block = [i for i, blk in enumerate(model.blocks) if blk.kind == "G"]
    if not gdn:
        die("model has no GatedDeltaNet layers")
    if not all(hasattr(m, "record_gate") for m in gdn):
        die("Train/model.py GatedDeltaNet has no record_gate interface")
    H = mc.gdn_heads
    pool = gdn[0].pool
    bound = float(mc.score_bound)
    warnings = []

    # per-layer accumulators, filled by a forward hook on each GatedDeltaNet right after it runs (frees last_gate early)
    A = [Moments(H, 0.0, 1.0, NBINS, device, sources) for _ in gdn]
    L = [Moments(H, *L1MA, device, sources) for _ in gdn]
    Bt = [Moments(H, 0.0, 1.0, NBINS, device, sources) for _ in gdn]
    S = [Moments(H, -bound, bound, NBINS, device, sources) for _ in gdn] if pool else None
    win = [[[] for _ in range(4)] for _ in gdn]
    gam_seen = [None] * len(gdn)
    state = {"source": None, "win_row": None, "guard": False}

    def hook(li):
        def f(mod, inp, outp):
            lg = mod.last_gate
            mod.last_gate = None
            a, b = lg["a"][0], lg["b"][0]          # (N, H): with cu the batch is flattened to (1, N, H)
            if state["guard"]:
                mod._eval_guard = {"a": lg["a"], "b": lg["b"]}
            la = log_alpha(a, mod.A_log, mod.dt_bias)
            al = torch.exp(la)
            src = state["source"]
            A[li].add(al, src, keep=la)            # quantiles taken on log alpha (monotone, exact near 1), mapped back
            L[li].add(torch.log10((-torch.expm1(la)).clamp_min(1e-30)), src)
            Bt[li].add(beta_of(b), src)
            if pool:
                g = lg["gamma"].float()
                gam_seen[li] = g.detach().cpu()
                s = lg["s"][0].float() if lg["s"] is not None else None
                if s is not None:
                    S[li].add(s, src)
                r = state["win_row"]
                if r is not None:
                    cu = inp[1]
                    n = a.shape[0]
                    seg_all = torch.zeros(n, dtype=torch.long, device=a.device)
                    seg_all[cu[1:-1].long()] = 1
                    seg_all = seg_all.cumsum(0)
                    sl = slice(r * T, (r + 1) * T)
                    s_row = s[sl] if s is not None else torch.zeros(T, H, device=a.device)
                    for acc, v in zip(win[li], window_stats(s_row, g, seg_all[sl], H)):
                        acc.append(v)
        return f

    handles = [m.register_forward_hook(hook(li)) for li, m in enumerate(gdn)]
    hidden = {}
    handles.append(model.norm.register_forward_hook(lambda mod, i, o: hidden.__setitem__("h", o)))
    for m in gdn:
        m.record_gate = True

    W = model.embed.weight.to(torch.bfloat16)
    edges = [0] + [2 ** i for i in range(int(math.log2(T)))] + [T]
    nb = len(edges) - 1
    ppl, pos = {}, {}
    max_dev = 0.0
    first = True
    with torch.no_grad():
        for name in sources:
            mm, k = val_rows(name, T)
            if args.max_rows:
                k = min(k, args.max_rows - args.max_rows % MB) or MB
            state["source"] = name
            fl_sum = tw_sum = 0.0
            n_tok = 0
            acc = {v: np.zeros((2, nb)) for v in ("doc", "context")}
            eot_acc = np.zeros(2)
            for bi, r0 in enumerate(range(0, k, MB)):
                buf = np.stack([mm[o * T:o * T + T + 1] for o in range(r0, r0 + MB)]).astype(np.int64)
                x, y, cu, cu_cpu = tensors(buf, eot, device)
                state["win_row"] = 0 if (pool and bi < WIN_ROWS) else None
                state["guard"] = first
                if first:
                    with KernelGuard() as kg, torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(x, y, cu, cu_cpu)
                    kg.check(gdn)
                    for m in gdn:
                        del m._eval_guard
                    first = False
                    state["guard"] = False
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(x, y, cu, cu_cpu)
                # per-token loss from the final hidden state, same bf16 operands as the fused loss in LM.forward
                h = hidden.pop("h")[0].to(torch.bfloat16)
                yf = y.reshape(-1)
                tok = torch.empty(yf.numel(), dtype=torch.float32, device=device)
                for c0 in range(0, yf.numel(), LOGIT_CHUNK):
                    lgt = F.linear(h[c0:c0 + LOGIT_CHUNK], W).float()
                    tok[c0:c0 + LOGIT_CHUNK] = F.cross_entropy(lgt, yf[c0:c0 + LOGIT_CHUNK], reduction="none")
                del lgt, h
                n = yf.numel()
                lf = float(loss.float())
                fl_sum += lf * n
                tw = float(tok.double().sum())
                tw_sum += tw
                n_tok += n
                max_dev = max(max_dev, abs(tw / n - lf))
                tok = tok.cpu().numpy().astype(np.float64)
                p, true_doc = positions(cu_cpu, n, T, r0, mm, eot)
                is_eot = buf[:, :-1].reshape(-1) == eot
                b = np.searchsorted(edges, p, side="right") - 1
                for v, sel in (("context", ~is_eot), ("doc", true_doc & ~is_eot)):
                    acc[v][0] += np.bincount(b[sel], weights=tok[sel], minlength=nb)
                    acc[v][1] += np.bincount(b[sel], minlength=nb)
                eot_acc += [tok[is_eot].sum(), is_eot.sum()]
            lo = fl_sum / n_tok
            ppl[name] = {"loss": lo, "ppl": math.exp(lo), "tokens": n_tok, "rows": k,
                         "loss_tokenwise": tw_sum / n_tok}
            pos[name] = {v: {"tokens": a[1].astype(int).tolist(), "loss": (a[0] / np.maximum(a[1], 1)).tolist()}
                         for v, a in acc.items()}
            pos[name]["eot_input"] = {"tokens": int(eot_acc[1]), "loss": float(eot_acc[0] / max(eot_acc[1], 1))}
            pos[name]["_sums"] = acc
            print(f"[evaluate] {name} loss {lo:.4f} ppl {math.exp(lo):.2f} tokens {n_tok}", flush=True)
    for hd in handles:
        hd.remove()
    for m in gdn:
        m.record_gate = False
    if max_dev > 2e-2:
        warnings.append(f"per-token loss deviates from the fused training loss by up to {max_dev:.4f} per batch")

    mean_loss = float(np.mean([ppl[s]["loss"] for s in sources]))
    dump(out, "ppl.json", {
        "run": run.name, "ckpt": ckpt.name, "step": step,
        "method": f"val.bin rows of {T}+1 tokens at offsets i*{T}, {MB} rows/batch, doc mask as in training; "
                  "loss = token-weighted mean of LM.forward's fused CE (same call as Train/train.py evaluate / old ppl.py); "
                  "loss_tokenwise = mean of per-token CE recomputed from the final hidden state (used for position buckets)",
        "per_source": ppl, "val_mean": mean_loss, "max_batch_dev_tokenwise_vs_fused": max_dev}, warnings)

    allp = {}
    for v in ("doc", "context"):
        sm = sum(pos[s]["_sums"][v] for s in sources)
        allp[v] = {"tokens": sm[1].astype(int).tolist(), "loss": (sm[0] / np.maximum(sm[1], 1)).tolist()}
    et = sum(pos[s]["eot_input"]["tokens"] for s in sources)
    allp["eot_input"] = {"tokens": et, "loss": sum(pos[s]["eot_input"]["loss"] * pos[s]["eot_input"]["tokens"]
                                                  for s in sources) / max(et, 1)}
    for s in sources:
        del pos[s]["_sums"]
    dump(out, "position_loss.json", {
        "run": run.name, "ckpt": ckpt.name, "step": step, "bucket_edges": edges,
        "definition": "bucket j holds predictions whose input token has position p in [edges[j], edges[j+1]); p = number "
                      "of tokens of its document before it (p = 0: the document's first token, predicting the second). "
                      "'doc': only doc-mask segments that begin at a true document start (after an eot, or a row start "
                      f"where val.bin[offset-1] is eot or offset 0); documents already running when the {T}-token row "
                      "starts are excluded because their true position exceeds the context the model sees. 'context': "
                      "every segment, p counted from the segment start (= row start for those continuations), i.e. "
                      "context length available to the model. Predictions whose input is the eot token (predicting the "
                      "next document's first token) are excluded from both and reported as 'eot_input'.",
        "per_source": pos, "all": allp}, warnings)

    gates = {"run": run.name, "ckpt": ckpt.name, "step": step, "gate_mode": mc.gate_mode,
             "formula": "alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias)), softplus(x) = x for x >= 20; "
                        "beta_t = sigmoid(b_t); a_t, b_t = raw gate logits passed to fla chunk_gated_delta_rule "
                        "(use_gate_in_kernel, use_beta_sigmoid_in_kernel); see comments in Eval/evaluate.py",
             "data": "all validation tokens of ppl.json; mean/std/hist over every token, quantiles over every "
                     f"{SAMPLE_STRIDE}th token; per_source_mean over every token",
             "layers": []}
    for li, m in enumerate(gdn):
        heads = []
        for h in range(H):
            heads.append({"head": h,
                          "alpha": A[li].head(h, qmap=torch.exp),
                          "log10_one_minus_alpha": L[li].head(h),
                          "beta": Bt[li].head(h),
                          "A_log": float(m.A_log[h]), "dt_bias": float(m.dt_bias[h])})
        gates["layers"].append({"gdn_layer": li, "block": gdn_block[li], "heads": heads})
    dump(out, "gates.json", gates, warnings)

    if pool:
        pj = {"run": run.name, "ckpt": ckpt.name, "step": step, "gate_mode": mc.gate_mode,
              "definition": "w_{t,i} = softmax over i<=t in the same doc-mask segment of (s_i - gamma (t-i)); "
                            "horizon = 1/(1-exp(-gamma)) (null when gamma = 0); score = bounded s = "
                            f"{bound}*tanh(raw/{bound}) from last_gate['s'] (absent for pool_s0, s = 0). window stats on "
                            f"row 0 of the first {WIN_ROWS} batches per source, all positions of those rows: ess = "
                            "1/sum w^2, exp_entropy = exp(-sum w log w), lookback = sum w (t-i), ctx = tokens available "
                            f"(t - segment start + 1); '_ctx_ge_{WIN_MIN_CTX}' restricts to positions with ctx >= "
                            f"{WIN_MIN_CTX} (window not cut by a document start).",
              "layers": []}
        for li, m in enumerate(gdn):
            g = gam_seen[li]
            ess, ent, look, ctx = (np.concatenate(v, -1) if v else np.zeros((H, 0)) for v in win[li])
            ctx = ctx.reshape(-1) if ctx.ndim == 1 else ctx
            heads = []
            for h in range(H):
                gh = float(g[h])
                d = {"head": h, "gamma": gh, "gamma_learned": m.gamma_raw is not None,
                     "horizon": None if gh <= 0 else float(1.0 / -math.expm1(-gh))}
                if S is not None and S[li].sample:
                    d["score"] = S[li].head(h)
                    if m.score_w is not None:
                        d["score_w_norm"] = float(m.score_w[h].float().norm())
                else:
                    d["score"] = None
                big = ctx >= WIN_MIN_CTX
                for key, arr in (("ess", ess), ("exp_entropy", ent), ("lookback", look)):
                    d[key] = plain_stats(arr[h])
                    d[f"{key}_ctx_ge_{WIN_MIN_CTX}"] = plain_stats(arr[h][big])
                heads.append(d)
            pj["layers"].append({"gdn_layer": li, "block": gdn_block[li], "heads": heads,
                                 "ab_c_norm": float(m.ab_c.float().norm()), "ab_x_norm": float(m.ab.weight.float().norm()),
                                 "window_ctx": plain_stats(ctx)})
        dump(out, "pool.json", pj, warnings)

    params = param_counts(model, gdn)
    tr = train_log(run, params["total"], cfg)
    dump(out, "train.json", {"run": run.name, "params": params, **tr}, warnings)

    secs = time.time() - t0
    mem = torch.cuda.max_memory_allocated(device) / 2 ** 30
    summary = {
        "run": run.name, "arm": ARMS.get(mc.gate_mode, mc.gate_mode), "gate_mode": mc.gate_mode,
        "seed": cfg["data"].get("seed"), "ckpt": ckpt.name, "step": step, "params": params["total"],
        "params_pool_extra": params["pool_extra"],
        "val_loss": {s: ppl[s]["loss"] for s in sources}, "val_ppl": {s: ppl[s]["ppl"] for s in sources},
        "val_mean_loss": mean_loss, "val_tokens_per_source": {s: ppl[s]["tokens"] for s in sources},
        "tok_s_mean": tr.get("tok_s_mean"), "final_train_loss": tr.get("final_train_loss"),
        "final_logged_val_mean": tr["val_curve"][-1]["val_mean"] if tr.get("val_curve") else None,
        "eval_seconds": secs, "eval_max_mem_gb": mem, "warnings": warnings,
    }
    if args.seq_len:
        summary["seq_len"] = T
        summary["train_seq_len"] = int(cfg["data"]["seq_len"])
    if args.max_rows:
        summary["partial"] = f"--max-rows {args.max_rows}: not a full evaluation"
    for w in warnings:
        print(f"[evaluate] WARNING: {w}", file=sys.stderr, flush=True)
    dump(out, "summary.json", summary, warnings)
    print(f"[evaluate] done {out}  val_mean {mean_loss:.4f}  {secs:.0f}s  max mem {mem:.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (model/<run name>)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default=None, help="checkpoint file name in the run dir or a path (default: final step_*.pt, else latest.pt)")
    ap.add_argument("--out", default=str(ROOT / "results"), help="results root (default research-llm/results)")
    ap.add_argument("--max-rows", type=int, default=0, help="testing only: limit rows per source (multiple of 8)")
    ap.add_argument("--seq-len", type=int, default=0, help="evaluation row length (default: config.toml data.seq_len); "
                    "position buckets extend by powers of two up to it")
    args = ap.parse_args()
    try:
        evaluate(args)
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        die(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
