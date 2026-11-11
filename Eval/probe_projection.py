"""Learned-projection probe: do contexts separate after a small learned MLP on frozen features?

Usage (from the research-llm directory):
    .venv/bin/python Eval/probe_projection.py --run model/plain_20261110_231750 --device cuda:0 [--steps 3000]
        -> results/probe/{probe.json, probe.pt, summary.md}

Frozen LM (loaded as Eval/switch_eval.py does), never trained. Data: data/switch (Train/build_switch_dataset.py):
2048-token rows = 2047 content tokens + eot; we feed the 2047 content tokens, one row per cu_seqlens sequence, no state
reset inside a row. Segment boundaries / documents from segments_{train,val}.npz. Train rows -> probe training,
val rows -> evaluation.

Feature x_t = input of the GatedDeltaNet mixer of layer l (= RMSNorm(residual), d_model 768), no_grad.
Probe (one per layer, all trained together): z_t = L2norm(W2 silu(W1 x_t + b1) + b2), 768 -> 256 -> 128.
Span summary = L2norm(mean of z over the span's tokens), true boundaries.
Loss: InfoNCE with learned temperature. Query = z_t, t in a returning segment (single token). Candidates = summaries of
spans of the same row that finished before t (span end <= t) + all spans of the other rows in the batch. Positives =
candidates of the same document (doc id). loss = logsumexp(valid) - logsumexp(positives).

Evaluation on val rows for three representations per layer: probe (trained), random_mlp (same architecture, untrained,
another seed), raw_k (k of that layer's GatedDeltaNet as the kernel uses it, l2norm per head; head-mean of per-head
cosines, summaries = per-head L2norm of the span mean of k, as in verify_structure M1).
  1. selection: query at t in a returning segment; candidates = finished spans of the same row (in_row), optionally
     + 6 distractor spans from other val rows (other documents than the row's; fixed per row). Correct if the best
     candidate is a same-document span. chance = mean of n_pos / n_cand. Query variants: single (z_t), segavg (causal
     mean of z over the current segment so far, oracle boundary), ema8 (causal EMA horizon 8 over the row).
     Buckets of the position in the returning segment: 0, 1-3, 4-15, 16-63, 64+.
  2. boundary detection (no true boundaries at query time): signals z64 = cos(z_t, EMA64(z)), s{2,4,8} =
     cos(EMA_s(z), EMA64(z)); EMA_h: y_t = r y_{t-1} + x_t, r = 1 - 1/h (verify_structure.ema). AUROC of -signal for
     "boundary in the last 32 tokens" over tokens t >= 32; detector, thresholds and metrics as verify_structure M1a.
  3. separation: cosine of summaries, same-document pairs (same row, same slot) vs different-document pairs in the
     same row and in other rows (20000 random pairs, different doc id); d' = (mean_same - mean_diff) / pooled std.
"""
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Eval"))
import verify_structure as V  # noqa: E402  (imports switch_eval -> evaluate -> Train/model.py)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

S, E, M = V.S, V.E, V.M
DATA = ROOT / "data" / "switch"
OUT = ROOT / "results" / "probe"
T_ROW = 2048
T = T_ROW - 1
D_IN, D_HID, D_Z = 768, 256, 128
POS_EDGES = [0, 1, 4, 16, 64]
POS_NAMES = ["0", "1-3", "4-15", "16-63", "64+"]
SHORTS, LONG = (2, 4, 8), 64
SIGS = ["z64"] + [f"s{h}" for h in SHORTS]
EMA_Q = 8
NDIST = 6
NPAIR = 20000
METHODS = ("probe", "random_mlp", "raw_k")
QVARS = ("single", "segavg", "ema8")
CSETS = ("in_row", f"plus{NDIST}")
WIN, QLEV = V.WIN, V.QLEV
LN100 = math.log(100.0)


def log(msg):
    print(f"[probe] {msg}", flush=True)


# ================================================================ data
class Split:
    def __init__(self, split):
        self.bin = np.memmap(DATA / "switch" / f"{split}.bin", dtype=np.uint16, mode="r")
        sg = dict(np.load(DATA / f"segments_{split}.npz"))
        self.ptr = sg["seg_ptr"]
        self.n = len(self.ptr) - 1
        self.start = sg["seg_start"].astype(np.int64)
        self.len = sg["seg_len"].astype(np.int64)
        self.slot = sg["seg_slot"].astype(np.int64)
        self.ret = sg["seg_return"]
        self.doc = sg["use_doc_id"][sg["seg_use"]]
        self.row = np.repeat(np.arange(self.n), np.diff(self.ptr))
        assert self.bin.shape[0] >= self.n * T_ROW

    def tokens(self, rows):
        return np.stack([np.asarray(self.bin[r * T_ROW:r * T_ROW + T], dtype=np.int64) for r in rows])

    def meta(self, rows, device):
        """Segments of the given rows, in-batch coordinates. Returns dict of device tensors."""
        segs = np.concatenate([np.arange(self.ptr[r], self.ptr[r + 1]) for r in rows])
        brow = np.concatenate([np.full(self.ptr[r + 1] - self.ptr[r], i) for i, r in enumerate(rows)])
        st, ln = self.start[segs], self.len[segs]
        seg_id = np.repeat(np.arange(len(segs)), ln)                          # per token, rows are contiguous
        assert len(seg_id) == len(rows) * T
        tok_pos = np.concatenate([np.arange(a, a + l) for a, l in zip(st, ln)])
        tok_row = brow[seg_id]
        tok_ret = self.ret[segs][seg_id]
        q = np.flatnonzero(tok_ret)
        g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        return {"segs": segs, "nS": len(segs), "seg_id": g(seg_id), "s_row": g(brow), "s_end": g(st + ln),
                "s_start": g(st), "s_len": g(ln), "s_doc": g(self.doc[segs]),
                "q_idx": g(q), "q_row": g(tok_row[q]), "q_t": g(tok_pos[q]), "q_doc": g(self.doc[segs][seg_id[q]]),
                "q_pos": g(tok_pos[q] - st[seg_id[q]]), "q_seg": g(seg_id[q])}


# ================================================================ frozen features
class Feats:
    def __init__(self, model, layers):
        self.model, self.layers, self.x = model, layers, {}
        for li in layers:
            model.blocks[li].mixer.register_forward_pre_hook(self._hook(li))

    def _hook(self, li):
        def f(mod, args):
            self.x[li] = args[0][0].to(torch.bfloat16)
        return f

    @torch.no_grad()
    def __call__(self, tok, device):
        B = tok.shape[0]
        cu_cpu = torch.arange(B + 1, dtype=torch.int64) * T
        cu = cu_cpu.to(device)
        x = torch.from_numpy(tok.reshape(-1)).to(device)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.model.embed(x)
            for li, blk in enumerate(self.model.blocks):
                h = blk(h, cu, cu_cpu)
                if li == max(self.layers):
                    break
        X = torch.stack([self.x[li] for li in self.layers])                       # (nL, N, 768) bf16
        self.x = {}
        return X, cu, cu_cpu


@torch.no_grad()
def raw_k(model, li, x, cu, cu_cpu):
    mx = model.blocks[li].mixer
    with torch.autocast("cuda", dtype=torch.bfloat16):
        qkv, _ = M.causal_conv1d(x=mx.qkv(x[None]), weight=mx.conv, activation="silu", backend="triton",
                                 cu_seqlens=cu, cu_seqlens_cpu=cu_cpu)
    _, k, _ = qkv.unflatten(-1, (3, mx.h, mx.dk)).unbind(-3)
    return V.l2n(k.float())[0]                                                      # (N,H,D)


# ================================================================ probe
class Probe(torch.nn.Module):
    """nL independent MLPs 768 -> 256 -> SiLU -> 128, nn.Linear default init (U(+-1/sqrt(fan_in)))."""

    def __init__(self, nL, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        U = lambda shape, fan: torch.nn.Parameter((torch.rand(shape, generator=g) * 2 - 1) / math.sqrt(fan))
        self.W1, self.b1 = U((nL, D_IN, D_HID), D_IN), U((nL, 1, D_HID), D_IN)
        self.W2, self.b2 = U((nL, D_HID, D_Z), D_HID), U((nL, 1, D_Z), D_HID)
        self.logit_scale = torch.nn.Parameter(torch.full((nL,), math.log(1 / 0.07)))

    def forward(self, X):
        h = F.silu(torch.baddbmm(self.b1, X, self.W1))
        return F.normalize(torch.baddbmm(self.b2, h, self.W2), dim=-1)


def summaries(R, seg_id, nS):
    """R (..., N, H, D) -> per-head L2norm of span means (..., nS, H, D)."""
    shp = R.shape[:-3] + (nS,) + R.shape[-2:]
    s = torch.zeros(shp, device=R.device, dtype=R.dtype).index_add_(R.dim() - 3, seg_id, R)
    return F.normalize(s, dim=-1)


def masks(mt):
    same = mt["q_row"][:, None] == mt["s_row"][None]
    valid = ~same | (mt["s_end"][None] <= mt["q_t"][:, None])
    pos = valid & (mt["q_doc"][:, None] == mt["s_doc"][None])
    return valid, pos


def info_nce(probe, X, mt):
    z = probe(X)                                                                     # (nL,N,Dz)
    Sm = summaries(z[..., None, :], mt["seg_id"], mt["nS"])[..., 0, :]              # (nL,nS,Dz)
    q = z[:, mt["q_idx"]]
    scale = probe.logit_scale.clamp(max=LN100).exp()[:, None, None]
    lg = torch.einsum("lqd,lsd->lqs", q, Sm) * scale
    valid, pos = masks(mt)
    ninf = torch.finfo(lg.dtype).min
    loss = torch.logsumexp(lg.masked_fill(~valid, ninf), -1) - torch.logsumexp(lg.masked_fill(~pos, ninf), -1)
    return loss.mean(-1)                                                             # (nL,)


def train(args, model, feats, layers, device):
    tr = Split("train")
    rng = np.random.default_rng([args.seed, 1])
    order = rng.permutation(tr.n)[:args.steps * args.batch]
    probe = Probe(len(layers), args.seed).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    hist, run_l, t0 = [], torch.zeros(len(layers), device=device), time.time()
    for step in range(args.steps):
        rows = order[step * args.batch:(step + 1) * args.batch]
        X, _, _ = feats(tr.tokens(rows), device)
        mt = tr.meta(rows, device)
        loss = info_nce(probe, X.float(), mt)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        sched.step()
        run_l += loss.detach()
        if (step + 1) % 100 == 0:
            l = (run_l / 100).tolist()
            hist.append({"step": step + 1, "loss": l, "temperature": (1 / probe.logit_scale.detach().clamp(
                max=LN100).exp()).tolist()})
            run_l.zero_()
            log(f"step {step + 1}/{args.steps} {time.time() - t0:.0f}s loss " + " ".join(f"{v:.3f}" for v in l))
    return probe, hist


# ================================================================ eval helpers
def ema(x, h, C=64):
    """Causal EMA along dim 1 of x (B,T,...): y_t = r*y_{t-1} + x_t, r = 1 - 1/h (chunked, float64 inside)."""
    B, Tn = x.shape[:2]
    r = 1.0 - 1.0 / h
    pad = (-Tn) % C
    xd = x.double()
    if pad:
        xd = torch.cat([xd, xd.new_zeros((B, pad) + tuple(x.shape[2:]))], 1)
    nC = xd.shape[1] // C
    xr = xd.view(B, nC, C, *x.shape[2:])
    j = torch.arange(C, device=x.device, dtype=torch.float64)
    shp = (1, 1, C) + (1,) * (x.dim() - 2)
    loc = torch.cumsum(xr * (r ** -j).view(shp), 2) * (r ** j).view(shp)
    pw = (r ** (j + 1)).view((1, C) + (1,) * (x.dim() - 2))
    carry = torch.zeros_like(loc[:, 0, -1])
    outs = []
    for c in range(nC):
        yc = loc[:, c] + pw * carry[:, None]
        outs.append(yc)
        carry = yc[:, -1]
    return torch.cat(outs, 1)[:, :Tn].float()


def hcos(a, b):
    """(..., H, D) x (..., H, D) -> mean over heads of per-head cosine."""
    return V.cosd(a, b).mean(-1)


def signals(R, B):
    """R (N,H,D) -> (len(SIGS), B, T) boundary signals (low = boundary)."""
    Rb = R.reshape(B, T, *R.shape[1:])
    e64 = ema(Rb, LONG)
    out = [hcos(Rb, e64)] + [hcos(ema(Rb, h), e64) for h in SHORTS]
    return torch.stack(out)


def qvariants(R, mt, B):
    """Query vectors (Nq,H,D) for single / segavg / ema8."""
    qi = mt["q_idx"]
    cs = torch.cumsum(R.double(), 0)
    cs = torch.cat([cs.new_zeros((1,) + tuple(R.shape[1:])), cs], 0)
    seg_first = mt["q_seg"]
    row_off = mt["q_row"] * T
    a = row_off + mt["s_start"][seg_first]
    segavg = (cs[qi + 1] - cs[a]).float()
    e8 = ema(R.reshape(B, T, *R.shape[1:]), EMA_Q).reshape(R.shape)[qi]
    return {"single": R[qi], "segavg": F.normalize(segavg, dim=-1), "ema8": F.normalize(e8, dim=-1)}


def pick_distractors(va, n, seed):
    rng = np.random.default_rng([seed, 3])
    out = np.zeros((n, NDIST), np.int64)
    nseg = int((va.row < n).sum())
    for r in range(n):
        docs = set(va.doc[va.ptr[r]:va.ptr[r + 1]].tolist())
        got = []
        while len(got) < NDIST:
            s = int(rng.integers(nseg))
            if va.row[s] == r or va.doc[s] in docs or s in got:
                continue
            got.append(s)
        out[r] = got
    return out


# ================================================================ boundary detection (verify_structure M1a, any #boundaries)
@torch.no_grad()
def detect(X, bnd, device):
    """X (Sg, n, T) signal on device (low = boundary); bnd (n, maxB) boundary starts, -1 pad."""
    Sg, n, Tn = X.shape
    maxB = bnd.shape[1]
    t = torch.arange(Tn, device=device)
    bt = torch.from_numpy(bnd).to(device)
    win = torch.full((n, Tn), -1, device=device, dtype=torch.long)
    for j in range(maxB):
        b = bt[:, j:j + 1]
        win = torch.where((b >= 0) & (t[None] >= b) & (t[None] < b + WIN), j, win)
    validt = (t >= WIN)[None].expand(n, Tn)
    lab = (win >= 0)[validt]
    nvalid = int(validt.sum())
    nbnd = int((bnd >= 0).sum())
    Xv = X[:, validt]
    au = V.auroc(-Xv, lab)
    m_in = Xv[:, ~lab].mean(1).cpu().numpy()
    m_af = Xv[:, lab].mean(1).cpu().numpy()
    ql = torch.tensor(QLEV, device=device, dtype=Xv.dtype)
    thr = torch.stack([torch.quantile(x, ql) for x in Xv])
    del Xv
    Q = thr.shape[1]
    last = torch.full((Sg, Q, n), -10 ** 6, device=device, dtype=torch.long)
    prev = torch.ones((Sg, Q, n), device=device, dtype=torch.bool)
    fa = torch.zeros((Sg, Q), device=device)
    non = torch.zeros((Sg, Q), device=device)
    first = torch.full((Sg, Q, n, maxB), -1, device=device, dtype=torch.long)
    btc = bt.clamp_min(0)
    for tt in range(WIN, Tn):
        below = X[:, None, :, tt] < thr[:, :, None]
        onset = below & ~prev & (tt - last >= WIN)
        last = torch.where(onset, tt, last)
        prev = below
        w = win[:, tt]
        fa += (onset & (w < 0)).sum(-1)
        non += onset.sum(-1)
        if bool((w >= 0).any()):
            wc = w.clamp_min(0)
            idx = wc.view(1, 1, n, 1).expand(Sg, Q, n, 1)
            cur = first.gather(-1, idx).squeeze(-1)
            setm = onset & (w >= 0) & (cur < 0)
            newv = torch.where(setm, tt - btc.gather(1, wc[:, None]).squeeze(1), cur)
            first.scatter_(-1, idx, newv[..., None])
    rate = (first >= 0).float().sum((-1, -2)) / nbnd
    prec = (non - fa) / non.clamp_min(1)
    f1 = 2 * prec * rate / (prec + rate).clamp_min(1e-12)
    far = fa / nvalid * 1000
    bi = f1.argmax(1)
    ok = far <= 1.0
    bi1 = torch.where(ok, rate, -1.0).argmax(1)
    firstc, ratec, farc, f1c = first.cpu().numpy(), rate.cpu().numpy(), far.cpu().numpy(), f1.cpu().numpy()
    okc = ok.any(1).cpu().numpy()

    def at(s, q):
        d = firstc[s, q][firstc[s, q] >= 0]
        return {"detection_rate": float(ratec[s, q]), "fa_per_1000": float(farc[s, q]), "quantile": float(QLEV[q]),
                "median_delay": float(np.median(d)) if len(d) else None, "f1": float(f1c[s, q])}
    return [{"auroc": float(au[s]), "mean_within": float(m_in[s]), "mean_after_boundary": float(m_af[s]),
             "best_f1": at(s, int(bi[s])), "best_at_fa_le_1": at(s, int(bi1[s])) if okc[s] else None}
            for s in range(Sg)]


# ================================================================ evaluation
@torch.no_grad()
def evaluate(args, model, feats, layers, probe, rprobe, device):
    va = Split("val")
    rows_all = np.arange(va.n if args.eval_rows <= 0 else min(args.eval_rows, va.n))
    n = len(rows_all)
    nL = len(layers)
    batches = [rows_all[i:i + args.batch] for i in range(0, n, args.batch)]
    seg_mask = va.row < n
    nseg = int(seg_mask.sum())
    H = model.blocks[0].mixer.h
    dims = {"probe": (1, D_Z), "random_mlp": (1, D_Z), "raw_k": (H, model.blocks[0].mixer.dk)}
    allsum = {m: torch.zeros((nL, nseg) + dims[m], device=device) for m in METHODS}
    sig = {m: np.zeros((nL, len(SIGS), n, T), np.float32) for m in METHODS}

    def zs(X):
        Xf = X.float()
        return probe(Xf), rprobe(Xf)

    def reps(X, z, cu, cu_cpu, li):
        return {"probe": z[0][li][:, None], "random_mlp": z[1][li][:, None],
                "raw_k": raw_k(model, layers[li], X[li], cu, cu_cpu)}

    t0 = time.time()
    for bi, rows in enumerate(batches):                               # pass 1: summaries, signals
        X, cu, cu_cpu = feats(va.tokens(rows), device)
        mt = va.meta(rows, device)
        z = zs(X)
        for li in range(nL):
            for m, R in reps(X, z, cu, cu_cpu, li).items():
                allsum[m][li, torch.from_numpy(mt["segs"]).to(device)] = summaries(R, mt["seg_id"], mt["nS"])
                sig[m][li, :, rows[0]:rows[-1] + 1] = signals(R, len(rows)).cpu().numpy()
        if bi % 10 == 0:
            log(f"eval pass1 {bi + 1}/{len(batches)} {time.time() - t0:.0f}s")
    dist = pick_distractors(va, n, args.seed)
    nb = len(POS_EDGES)
    # acc[method, layer, qvar, cset, bucket]: correct, chance, count
    corr = np.zeros((len(METHODS), nL, len(QVARS), len(CSETS), nb))
    chan = np.zeros_like(corr)
    cnt = np.zeros(nb)
    for bi, rows in enumerate(batches):                               # pass 2: selection
        X, cu, cu_cpu = feats(va.tokens(rows), device)
        mt = va.meta(rows, device)
        z = zs(X)
        valid, pos = masks(mt)
        valid = valid & (mt["s_row"][None] == mt["q_row"][:, None])  # in-row candidates only
        pos = pos & valid
        npos, nval = pos.sum(-1).double(), valid.sum(-1).double()
        assert bool((npos > 0).all())
        bk = torch.bucketize(mt["q_pos"], torch.tensor(POS_EDGES[1:], device=device), right=True)
        cnt += np.bincount(bk.cpu().numpy(), minlength=nb)
        segs_t = torch.from_numpy(mt["segs"]).to(device)
        dist_t = torch.from_numpy(dist[rows]).to(device)                   # (B,NDIST) global seg idx
        for li in range(nL):
            for mi, (m, R) in enumerate(reps(X, z, cu, cu_cpu, li).items()):
                Cin = allsum[m][li, segs_t]                                # (nS,H,D)
                Cd = allsum[m][li, dist_t]                                 # (B,NDIST,H,D)
                for qi, (qn, Q) in enumerate(qvariants(R, mt, len(rows)).items()):
                    sc = torch.einsum("qhd,shd->qs", Q, Cin) / Q.shape[1]
                    scd = torch.einsum("qhd,bkhd->qbk", Q, Cd) / Q.shape[1]
                    scd = scd[torch.arange(len(Q), device=device), mt["q_row"]].amax(-1)
                    pmax = sc.masked_fill(~pos, -9.0).amax(-1)
                    nmax = sc.masked_fill(~(valid & ~pos), -9.0).amax(-1)
                    for ci, (c, ch) in enumerate((((pmax > nmax), npos / nval),
                                                  ((pmax > torch.maximum(nmax, scd)), npos / (nval + NDIST)))):
                        corr[mi, li, qi, ci] += torch.zeros(nb, device=device, dtype=torch.float64).index_add_(
                            0, bk, c.double()).cpu().numpy()
                        chan[mi, li, qi, ci] += torch.zeros(nb, device=device, dtype=torch.float64).index_add_(
                            0, bk, ch).cpu().numpy()
        if bi % 10 == 0:
            log(f"eval pass2 {bi + 1}/{len(batches)} {time.time() - t0:.0f}s")
    sel = {}
    for mi, m in enumerate(METHODS):
        sel[m] = {}
        for li, l in enumerate(layers):
            e = {}
            for qi, qn in enumerate(QVARS):
                for ci, c in enumerate(CSETS):
                    e[f"{qn}_{c}"] = {"by_bucket": (corr[mi, li, qi, ci] / cnt).tolist(),
                                      "overall": float(corr[mi, li, qi, ci].sum() / cnt.sum()),
                                      "chance_by_bucket": (chan[mi, li, qi, ci] / cnt).tolist(),
                                      "chance": float(chan[mi, li, qi, ci].sum() / cnt.sum())}
            sel[m][f"layer_{l}"] = e
    log(f"selection done {time.time() - t0:.0f}s")
    # ---- separation
    rows_seg, slot_seg, doc_seg = va.row[:nseg], va.slot[:nseg], va.doc[:nseg]
    same, diff_row = [], []
    for r in range(n):
        ids = np.arange(va.ptr[r], va.ptr[r + 1])
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                (same if slot_seg[ids[a]] == slot_seg[ids[b]] else diff_row).append((ids[a], ids[b]))
    rng = np.random.default_rng([args.seed, 4])
    other = []
    while len(other) < NPAIR:
        i, j = rng.integers(nseg, size=2)
        if rows_seg[i] != rows_seg[j] and doc_seg[i] != doc_seg[j]:
            other.append((i, j))
    pairs = {"same_doc": np.array(same), "diff_doc_same_row": np.array(diff_row), "diff_doc_other_row": np.array(other)}
    sep = {}
    for m in METHODS:
        sep[m] = {}
        for li, l in enumerate(layers):
            e = {}
            for pn, p in pairs.items():
                c = hcos(allsum[m][li, p[:, 0]], allsum[m][li, p[:, 1]]).cpu().numpy()
                e[pn] = {"mean": float(c.mean()), "std": float(c.std()), "n": int(len(c))}
            for dn in ("diff_doc_same_row", "diff_doc_other_row"):
                e[f"dprime_same_vs_{dn}"] = float((e["same_doc"]["mean"] - e[dn]["mean"])
                                                 / np.sqrt((e["same_doc"]["std"] ** 2 + e[dn]["std"] ** 2) / 2))
            sep[m][f"layer_{l}"] = e
    log(f"separation done {time.time() - t0:.0f}s")
    # ---- boundary detection
    nb_row = [va.start[va.ptr[r] + 1:va.ptr[r + 1]] for r in range(n)]
    maxB = max(len(b) for b in nb_row)
    bnd = np.full((n, maxB), -1, np.int64)
    for r, b in enumerate(nb_row):
        bnd[r, :len(b)] = b
    det = {m: {} for m in METHODS}
    for li, l in enumerate(layers):
        Xs = torch.from_numpy(np.concatenate([sig[m][li] for m in METHODS])).to(device)   # (3*nsig, n, T)
        res = detect(Xs, bnd, device)
        del Xs
        for mi, m in enumerate(METHODS):
            det[m][f"layer_{l}"] = {s: res[mi * len(SIGS) + si] for si, s in enumerate(SIGS)}
        log(f"detection layer {l} done {time.time() - t0:.0f}s")
    return {"rows": n, "spans": nseg, "boundaries": int((bnd >= 0).sum()), "query_tokens_by_bucket": cnt.tolist(),
            "selection": sel, "separation": sep, "boundary_detection": det}


# ================================================================ summary
def write_summary(res):
    L = res["layers"]
    ev = res["eval"]
    sel, sep, det = ev["selection"], ev["separation"], ev["boundary_detection"]
    lines = ["# Learned-projection probe on frozen features", "",
             f"Run `{res['run']}` ({res['ckpt']}), frozen. Feature = input of the GatedDeltaNet mixer (RMSNorm of the "
             f"residual). Probe 768-256-128 (SiLU), InfoNCE, {res['train']['steps']} steps x {res['train']['batch']} "
             f"train rows of data/switch, learned temperature. Eval on {ev['rows']} val rows ({ev['spans']} spans, "
             f"{ev['boundaries']} boundaries). Methods: probe = trained MLP, random = same MLP untrained, raw_k = "
             "per-head l2norm k of that layer (head-mean cosine), as verify_structure M1. Definitions: "
             "`Eval/probe_projection.py` docstring; all numbers in `results/probe/probe.json`.", "",
             "Final train loss (mean of last 100 steps) / temperature per layer: " + ", ".join(
                 f"L{l} {res['train']['history'][-1]['loss'][i]:.3f}/{res['train']['history'][-1]['temperature'][i]:.3f}"
                 for i, l in enumerate(L)), ""]
    short = {"probe": "probe", "random_mlp": "random", "raw_k": "raw_k"}
    for c, title in ((CSETS[0], "in-row candidates only"), (CSETS[1], f"in-row + {NDIST} distractors from other rows")):
        lines += [f"## 1. Selection top-1, {title} (overall over query tokens in returning segments)", "",
                  "| layer | method | single | segavg | ema8 | chance |", "|---|---|---|---|---|---|"]
        for l in L:
            for m in METHODS:
                e = sel[m][f"layer_{l}"]
                lines.append(f"| {l} | {short[m]} | " + " | ".join(f"{e[f'{q}_{c}']['overall']:.3f}" for q in QVARS)
                             + f" | {e[f'single_{c}']['chance']:.3f} |")
        lines.append("")
    for q in QVARS:
        for c in CSETS:
            if q != "single" and c == CSETS[0]:
                continue
            e0 = sel[METHODS[0]][f"layer_{L[0]}"][f"{q}_{c}"]
            lines += [f"### Selection by position in the returning segment, query {q}, {c}", "",
                      "| layer | method | " + " | ".join(POS_NAMES) + " |", "|---|---|" + "---|" * len(POS_NAMES),
                      "| - | chance | " + " | ".join(f"{v:.3f}" for v in e0["chance_by_bucket"]) + " |"]
            for l in L:
                for m in METHODS:
                    e = sel[m][f"layer_{l}"][f"{q}_{c}"]
                    lines.append(f"| {l} | {short[m]} | " + " | ".join(f"{v:.3f}" for v in e["by_bucket"]) + " |")
            lines.append("")
    lines += [f"Query tokens per bucket: " + ", ".join(f"{p} {int(v)}" for p, v in
                                                       zip(POS_NAMES, ev["query_tokens_by_bucket"])), ""]
    lines += ["## 2. Boundary detection without true boundaries", "",
              "AUROC of -signal for 'boundary in the last 32 tokens' (tokens t >= 32). z64 = cos(z_t, EMA64), "
              "sH = cos(EMA_H, EMA64).", "", "| layer | method | " + " | ".join(SIGS) + " |",
              "|---|---|" + "---|" * len(SIGS)]
    for l in L:
        for m in METHODS:
            e = det[m][f"layer_{l}"]
            lines.append(f"| {l} | {short[m]} | " + " | ".join(f"{e[s]['auroc']:.3f}" for s in SIGS) + " |")
    lines += ["", "Best signal per layer and method (by AUROC). det/FA/delay at the max-F1 threshold; det@FA<=1 and "
              "its median delay at the best threshold with <= 1 false alarm per 1000 tokens (verify_structure M1a).", "",
              "| layer | method | signal | AUROC | sig within | sig after bnd | det | FA/1k | delay | det@FA<=1 | "
              "delay@FA<=1 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in L:
        for m in METHODS:
            e = det[m][f"layer_{l}"]
            s = max(SIGS, key=lambda k: e[k]["auroc"])
            x = e[s]
            bf, d1 = x["best_f1"], x["best_at_fa_le_1"]
            lines.append(f"| {l} | {short[m]} | {s} | {x['auroc']:.3f} | {x['mean_within']:.3f} | "
                         f"{x['mean_after_boundary']:.3f} | {bf['detection_rate']:.2f} | {bf['fa_per_1000']:.2f} | "
                         f"{bf['median_delay']} | {(d1['detection_rate'] if d1 else float('nan')):.2f} | "
                         f"{d1['median_delay'] if d1 else None} |")
    lines += ["", "## 3. Summary separation (cosine of span summaries, mean ± std over pairs)", "",
              "| layer | method | same doc | diff doc same row | diff doc other row | d' same row | d' other row |",
              "|---|---|---|---|---|---|---|"]
    for l in L:
        for m in METHODS:
            e = sep[m][f"layer_{l}"]
            g = lambda k: f"{e[k]['mean']:.3f}±{e[k]['std']:.3f}"
            lines.append(f"| {l} | {short[m]} | {g('same_doc')} | {g('diff_doc_same_row')} | "
                         f"{g('diff_doc_other_row')} | {e['dprime_same_vs_diff_doc_same_row']:.2f} | "
                         f"{e['dprime_same_vs_diff_doc_other_row']:.2f} |")
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {OUT / 'summary.md'}")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="model/plain_20261110_231750")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--layers", default="all", help="'all' or comma list, e.g. 0,2,4,6,8,11")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_rows", type=int, default=0, help="0 = all val rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out:
        globals()["OUT"] = Path(args.out).resolve()
    try:
        t0 = time.time()
        OUT.mkdir(parents=True, exist_ok=True)
        run = Path(args.run).resolve()
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        model, cfg, ckpt, step = S.load_model(run, device, args.ckpt)
        for p in model.parameters():
            p.requires_grad_(False)
        nLm = len(model.blocks)
        layers = list(range(nLm)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
        log(f"run {run.name} ckpt {ckpt.name} step {step} layers {layers}")
        feats = Feats(model, layers)
        probe, hist = train(args, model, feats, layers, device)
        probe.eval()
        rprobe = Probe(len(layers), args.seed + 1000).to(device).eval()
        torch.save({"layers": layers, "probe": probe.state_dict(), "random_mlp": rprobe.state_dict()}, OUT / "probe.pt")
        t_train = time.time() - t0
        log(f"train done {t_train:.0f}s")
        ev = evaluate(args, model, feats, layers, probe, rprobe, device)
        res = {"run": run.name, "ckpt": ckpt.name, "step": step, "layers": layers, "feature": "mixer input (RMSNorm(x))",
               "train": {"steps": args.steps, "batch": args.batch, "lr": args.lr, "seed": args.seed,
                         "seconds": t_train, "history": hist},
               "eval": ev, "seconds": time.time() - t0}
        warnings = []
        E.dump(OUT, "probe.json", res, warnings)
        for w in warnings:
            log(f"WARNING: {w}")
        write_summary(res)
        log(f"done {time.time() - t0:.0f}s")
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        E.die(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
