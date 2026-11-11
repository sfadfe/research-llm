"""Training-free checks: is a redesigned attention (context-preserving / context-selecting) worth building?

Usage (from the research-llm directory):
    .venv/bin/python Eval/verify_structure.py --run model/<run dir> --device cuda:<N> [--part all|m3|m1|m2]
        -> results/verify/<run name>/{m3_preserve.json, m3_rows.npz, m1_pooling.json, m2_rank.json}
        (--part m2: M2 alone, uncentred + centred, rewrites m2_rank.json / m2_frac.npz; M1/M3 not recomputed)
    .venv/bin/python Eval/verify_structure.py --summarize model/<run A> model/<run B>
        -> results/verify/summary.md, results/verify/paired_m3.json (B - A per row)

Data: val.bin of the 5 switch_eval sources (Eval/switch_eval.py load_docs / Stream / toks / TokenLoss / load_model are
reused). Rows are packed with cu_seqlens only at row starts: nothing is reset inside a row (switch_eval no-reset mode).

M3 (preservation upper bound): A doc >= 1024 tokens, A1 = A[0,512), A2 = A[512,1024); B = first L tokens of a doc
(>= 2048 tokens) of another source, same B doc for every L in {64,512,2048}. (i) A1|B|A2, (ii) A1|A2, (iii) A2,
(iv) B|A2. Loss on A2 input positions p in [0,511) (target = next A2 token, switch_eval convention).

M1/M2 rows: A1|B|A2|C, A doc >= 1024, B, C docs >= 512 from two different sources other than A's. Fixed variant:
512-token segments; variable variant: each segment length uniform in [64,512]. A1 = A[0,l0), A2 = A[l0,l0+l2).
Per GatedDeltaNet layer we capture k as the kernel uses it (l2norm, eps 1e-6), v (after conv+silu) and beta =
sigmoid(b) (record_gate path, i.e. pool computed eagerly for pool arms), via a per-instance forward wrapper.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Eval"))
import switch_eval as S  # noqa: E402  (imports evaluate -> HIPBLASLT path, Train/model.py as S.M)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

E, M = S.E, S.M
SOURCES = S.SOURCES
SEG = 512
OUT = ROOT / "results" / "verify"

M3_BL = (64, 512, 2048)
M3_EDGES = [0, 16, 64, 256, 511]          # buckets over scored A2 positions p in [0,511)
SHORT, LONG = (8, 16), (64, 128, 256)
CFGS = [(s, l) for s in SHORT for l in LONG]
WIN = 32                                  # "boundary within the last 32 tokens"; also detector refractory period
SEL_H = 16
SEL_EDGES = [0, 16, 64, 256, 512]
NDIST = 6
M2_L = (64, 256, 512)
M2_M = (1, 2, 4, 8, 16, 32)
QLEV = np.geomspace(5e-4, 0.4, 48)        # threshold grid = these quantiles of each signal (valid tokens)
NPAIR = 20000
BATCH_TOK = 16384


def log(msg):
    print(f"[verify] {msg}", flush=True)


def akey(src, doc):
    return np.asarray(src, np.int64) * 1_000_000 + np.asarray(doc, np.int64)


# ================================================================ M3
def build_m3(docs, n, seed):
    rng = np.random.default_rng([seed, 31])
    okA = [np.flatnonzero(docs[s][2] >= 2 * SEG) for s in SOURCES]
    okB = [np.flatnonzero(docs[s][2] >= max(M3_BL)) for s in SOURCES]
    sa = [S.Stream(i, rng) for i in okA]
    sb = [S.Stream(i, rng) if len(i) else None for i in okB]
    a_src = np.arange(n) % len(SOURCES)
    b_src = np.empty(n, np.int64)
    for r, a in enumerate(a_src):
        cand = [k for k in range(len(SOURCES)) if k != a and len(okB[k])]
        b_src[r] = cand[(r // len(SOURCES)) % len(cand)]
    a_doc = np.array([sa[k].next() for k in a_src], np.int64)
    b_doc = np.array([sb[k].next() for k in b_src], np.int64)
    return a_src, a_doc, b_src, b_doc


def m3_conds():
    return ["ii", "iii"] + [f"i_{L}" for L in M3_BL] + [f"iv_{L}" for L in M3_BL]


@torch.no_grad()
def run_m3(tl, docs, a_src, a_doc, b_src, b_doc):
    R = len(a_src)
    A1 = lambda r: S.toks(docs, a_src[r], a_doc[r], 0, SEG)
    A2 = lambda r: S.toks(docs, a_src[r], a_doc[r], SEG, 2 * SEG)
    Bt = lambda r, L: S.toks(docs, b_src[r], b_doc[r], 0, L)
    mk = {"ii": lambda r: (np.concatenate([A1(r), A2(r)]), SEG), "iii": lambda r: (A2(r), 0)}
    for L in M3_BL:
        mk[f"i_{L}"] = (lambda L: lambda r: (np.concatenate([A1(r), Bt(r, L), A2(r)]), SEG + L))(L)
        mk[f"iv_{L}"] = (lambda L: lambda r: (np.concatenate([Bt(r, L), A2(r)]), L))(L)
    out = {}
    for c in m3_conds():
        res = np.zeros((R, SEG - 1), np.float32)
        built = [mk[c](r) for r in range(R)]
        for idx in S.packs([len(b[0]) for b in built]):
            o = tl([built[r][0] for r in idx], [[] for _ in idx])
            for k, r in enumerate(idx):
                res[r] = o[k][built[r][1]:built[r][1] + SEG - 1]
        out[c] = res
        log(f"m3 {c} done")
    return out


def bucketize(x, edges):
    """x (R, P) per position -> (R, nb + 1): bucket means and overall token mean."""
    cols = [x[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])]
    return np.column_stack(cols + [x.mean(1)])


def m3_entry(x, rows, keys):
    X = bucketize(x.astype(np.float64), M3_EDGES)
    return {"mean": X.mean(0).tolist(), "se_row": S.cluster_se(X, rows).tolist(),
            "se_doc": S.cluster_se(X, keys).tolist()}


def m3_stats(vals, a_src, a_doc):
    R = len(a_src)
    rows, keys = np.arange(R), akey(a_src, a_doc)
    st = {"levels": {c: m3_entry(v, rows, keys) for c, v in vals.items()}}
    d = {"iii_minus_ii": m3_entry(vals["iii"] - vals["ii"], rows, keys)}
    for L in M3_BL:
        d[f"i_minus_ii_L{L}"] = m3_entry(vals[f"i_{L}"] - vals["ii"], rows, keys)
        d[f"iv_minus_ii_L{L}"] = m3_entry(vals[f"iv_{L}"] - vals["ii"], rows, keys)
        d[f"iii_minus_i_L{L}"] = m3_entry(vals["iii"] - vals[f"i_{L}"], rows, keys)
    st["diffs"] = d
    return st


# ================================================================ M1 / M2 data
def build_m1(docs, n, seed, variable):
    rng = np.random.default_rng([seed, 11, int(variable)])
    okA = [np.flatnonzero(docs[s][2] >= 2 * SEG) for s in SOURCES]
    okO = [np.flatnonzero(docs[s][2] >= SEG) for s in SOURCES]
    sa = [S.Stream(i, rng) for i in okA]
    so = [S.Stream(i, rng) for i in okO]
    src = np.zeros((n, 4), np.int64)
    doc = np.zeros((n, 4), np.int64)
    for r in range(n):
        a = r % len(SOURCES)
        b, c = rng.choice([k for k in range(len(SOURCES)) if k != a], 2, replace=False)
        src[r] = [a, b, a, c]
        da = sa[a].next()
        doc[r] = [da, so[b].next(), da, so[c].next()]
    ln = rng.integers(64, SEG + 1, size=(n, 4)) if variable else np.full((n, 4), SEG)
    start = np.zeros((n, 4), np.int64)
    start[:, 2] = ln[:, 0]
    # distractor spans for selection: (row, seg) from other rows, not the row's A or B document
    keys = akey(src, doc)
    dist = np.zeros((n, NDIST, 2), np.int64)
    for r in range(n):
        got = []
        while len(got) < NDIST:
            r2, s2 = int(rng.integers(n)), int(rng.integers(4))
            if r2 == r or keys[r2, s2] in (keys[r, 0], keys[r, 1]) or (r2, s2) in got:
                continue
            got.append((r2, s2))
        dist[r] = got
    return {"src": src, "doc": doc, "start": start, "len": ln, "dist": dist}


def m1_row(docs, sp, r):
    return np.concatenate([S.toks(docs, sp["src"][r, j], sp["doc"][r, j], sp["start"][r, j],
                                  sp["start"][r, j] + sp["len"][r, j]) for j in range(4)])


# ================================================================ capture
def l2n(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


class Capture:
    """Per-instance forward wrapper on every GatedDeltaNet: stores (k l2-normed, v, beta) float32, shapes (N,H,D),
    (N,H,D), (N,H) for the flattened packed batch. model.py is not modified."""

    def __init__(self, model):
        self.model = model
        self.mix = [b.mixer for b in model.blocks if isinstance(b.mixer, M.GatedDeltaNet)]
        self.store = [None] * len(self.mix)
        for li, mx in enumerate(self.mix):
            mx.record_gate = True
            mx.forward = self._wrap(mx, mx.forward, li)

    def _wrap(self, mx, orig, li):
        def fwd(x, cu=None, cu_cpu=None):
            qkv, _ = M.causal_conv1d(x=mx.qkv(x), weight=mx.conv, activation="silu", backend="triton",
                                     cu_seqlens=cu, cu_seqlens_cpu=cu_cpu)
            _, k, v = qkv.unflatten(-1, (3, mx.h, mx.dk)).unbind(-3)
            out = orig(x, cu, cu_cpu)
            beta = torch.sigmoid(mx.last_gate["b"].float())
            self.store[li] = (l2n(k.float())[0], v.float()[0], beta.reshape(-1, mx.h))
            mx.last_gate = None
            return out
        return fwd

    @torch.no_grad()
    def __call__(self, rows, device):
        """rows: list of 1-D token arrays. Returns per layer list over rows of (k, v, beta)."""
        lens = [len(r) for r in rows]
        cu_cpu = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.int64)
        x = torch.from_numpy(np.concatenate(rows)).to(device)[None]
        cu = cu_cpu.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.model.embed(x)
            for blk in self.model.blocks:
                h = blk(h, cu, cu_cpu)
        out = []
        for k, v, b in self.store:
            out.append([(k[a:e], v[a:e], b[a:e]) for a, e in zip(cu_cpu[:-1].tolist(), cu_cpu[1:].tolist())])
        self.store = [None] * len(self.mix)
        return out


def ema(x, h):
    """Causal EMA along dim 0 with decay r = 1 - 1/h, up to a per-position scale (irrelevant for cosine). float64."""
    T = x.shape[0]
    r = 1.0 - 1.0 / h
    t = torch.arange(T, device=x.device, dtype=torch.float64)
    assert T * -np.log(r) < 700
    shp = (T,) + (1,) * (x.dim() - 1)
    return torch.cumsum(x.double() * (r ** -t).view(shp), 0) * (r ** t).view(shp)


def cosd(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


# ================================================================ M1 / M2 pass 1
@torch.no_grad()
def pass1(cap, docs, sp, device, do_m2):
    n = len(sp["src"])
    nL = len(cap.mix)
    H, D = cap.mix[0].h, cap.mix[0].dk
    tot = sp["len"].sum(1)
    Tmax = int(tot.max())
    sig = np.full((n, len(CFGS), nL, H + 1, Tmax), np.nan, np.float32)
    kmean = np.zeros((n, nL, 4, H, D), np.float32)
    vmean = np.zeros((n, nL, 4, H, D), np.float32)
    m2 = np.zeros((n, nL, 4, len(M2_L), H, len(M2_M)), np.float32) if do_m2 else None
    m2pr = np.zeros((n, nL, 4, len(M2_L), H), np.float32) if do_m2 else None
    mi = torch.tensor(M2_M) - 1
    t_svd = 0.0
    for idx in S.packs(list(tot + 1), BATCH_TOK):
        feats = cap([m1_row(docs, sp, r) for r in idx], device)
        for li in range(nL):
            for j, r in enumerate(idx):
                k, v, b = feats[li][j]
                T = k.shape[0]
                emas = {h: ema(k, h) for h in set(SHORT) | set(LONG)}
                for ci, (hs, hl) in enumerate(CFGS):
                    c = cosd(emas[hs], emas[hl]).float()          # (T,H)
                    sig[r, ci, li, :H, :T] = c.T.cpu().numpy()
                    sig[r, ci, li, H, :T] = c.mean(1).cpu().numpy()
                st = np.concatenate([[0], np.cumsum(sp["len"][r])])
                for s in range(4):
                    kmean[r, li, s] = k[st[s]:st[s + 1]].mean(0).cpu().numpy()
                    vmean[r, li, s] = v[st[s]:st[s + 1]].mean(0).cpu().numpy()
                if do_m2:
                    mats = []
                    for s in range(4):
                        for L in M2_L:
                            a = st[s]
                            bv = b[a:a + L, :, None] * v[a:a + L]
                            mats.append(torch.einsum("thd,the->hde", bv, k[a:a + L]))
                    t0 = time.time()
                    sv = torch.linalg.svdvals(torch.stack(mats).double().cpu()) ** 2  # (4*3, H, D) desc; CPU f64 (GPU svd slow)
                    t_svd += time.time() - t0
                    fr = sv.cumsum(-1) / sv.sum(-1, keepdim=True)
                    m2[r, li] = fr[..., mi].view(4, len(M2_L), H, len(M2_M)).float().cpu().numpy()
                    m2pr[r, li] = (sv.sum(-1) ** 2 / (sv ** 2).sum(-1)).view(4, len(M2_L), H).float().cpu().numpy()
        log(f"pass1 rows {idx[-1] + 1}/{n} (svd {t_svd:.0f}s)")
    return sig, kmean, vmean, m2, m2pr


# ================================================================ M2 alone (uncentred + centred)
def m2_mats(feats, li, j, lens):
    """sum_i beta_i v_i k_i^T over the first L tokens of each of the 4 segments -> (4*len(M2_L), H, D, D) float32."""
    k, v, b = feats[li][j]
    st = np.concatenate([[0], np.cumsum(lens)])
    mats = []
    for s in range(4):
        for L in M2_L:
            a = st[s]
            mats.append(torch.einsum("thd,the->hde", b[a:a + L, :, None] * v[a:a + L], k[a:a + L]))
    return torch.stack(mats)


@torch.no_grad()
def run_m2(cap, docs, sp, device):
    """Pass a: per layer/head mean of M/len over all spans (rows x 4 segments x span lengths). Pass b: SVD of M
    (uncentred) and of M_c = M - len * mean (centred)."""
    n = len(sp["src"])
    nL = len(cap.mix)
    H, D = cap.mix[0].h, cap.mix[0].dk
    tot = sp["len"].sum(1)
    Lvec = torch.tensor(np.tile(M2_L, 4), dtype=torch.float64)                    # length of each of the 12 mats
    msum = torch.zeros((nL, H, D, D), dtype=torch.float64, device=device)
    cnt = 0
    for idx in S.packs(list(tot + 1), BATCH_TOK):
        feats = cap([m1_row(docs, sp, r) for r in idx], device)
        for j, r in enumerate(idx):
            for li in range(nL):
                mt = m2_mats(feats, li, j, sp["len"][r]).double()
                msum[li] += (mt / Lvec.to(device).view(-1, 1, 1, 1)).sum(0)
            cnt += len(Lvec)
        log(f"m2 mean rows {idx[-1] + 1}/{n}")
    mbar = (msum / cnt).cpu()                                                     # (nL,H,D,D)
    shp = (n, nL, 4, len(M2_L))
    fr = np.zeros(shp + (H, len(M2_M)), np.float32)
    pr = np.zeros(shp + (H,), np.float32)
    frc, prc = np.zeros_like(fr), np.zeros_like(pr)
    shared = np.zeros(shp + (H,), np.float32)
    resid = np.zeros(shp + (H,), np.float32)
    mi = torch.tensor(M2_M) - 1
    t_svd = 0.0
    for idx in S.packs(list(tot + 1), BATCH_TOK):
        feats = cap([m1_row(docs, sp, r) for r in idx], device)
        for j, r in enumerate(idx):
            for li in range(nL):
                mt = m2_mats(feats, li, j, sp["len"][r]).double().cpu()            # (12,H,D,D)
                sh = Lvec.view(-1, 1, 1, 1) * mbar[li][None]
                mc = mt - sh
                nrm = (mt ** 2).sum((-1, -2))                                     # (12,H)
                shared[r, li] = ((sh ** 2).sum((-1, -2)) / nrm).view(4, len(M2_L), H).float().numpy()
                resid[r, li] = ((mc ** 2).sum((-1, -2)) / nrm).view(4, len(M2_L), H).float().numpy()
                t0 = time.time()
                sv = torch.linalg.svdvals(torch.stack([mt, mc])) ** 2                # (2,12,H,D) desc; CPU f64
                t_svd += time.time() - t0
                f = sv.cumsum(-1) / sv.sum(-1, keepdim=True)
                p = sv.sum(-1) ** 2 / (sv ** 2).sum(-1)
                fr[r, li] = f[0][..., mi].view(4, len(M2_L), H, len(M2_M)).float().numpy()
                frc[r, li] = f[1][..., mi].view(4, len(M2_L), H, len(M2_M)).float().numpy()
                pr[r, li] = p[0].view(4, len(M2_L), H).float().numpy()
                prc[r, li] = p[1].view(4, len(M2_L), H).float().numpy()
        log(f"m2 svd rows {idx[-1] + 1}/{n} (svd {t_svd:.0f}s)")
    return fr, pr, frc, prc, shared, resid


# ================================================================ M1a boundary detection
def auroc(score, label):
    """score (S, N) higher = boundary, label (N,) bool -> (S,) AUROC (ties broken arbitrarily)."""
    order = score.argsort(1)
    ranks = torch.empty_like(order)
    ar = torch.arange(score.shape[1], device=score.device).expand_as(order)
    ranks.scatter_(1, order, ar)
    ranks = ranks.double() + 1
    npos = label.sum().double()
    nneg = label.numel() - npos
    return ((ranks[:, label].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)).cpu().numpy()


def m1a(sig, sp, device):
    n, nC, nL, nH1, Tmax = sig.shape
    H = nH1 - 1
    ln = sp["len"]
    tot = ln.sum(1)
    bpos = np.cumsum(ln, 1)[:, :3]                                     # boundary starts of B, A2, C
    t = np.arange(Tmax)
    valid = (t[None] >= WIN) & (t[None] < tot[:, None])
    win = np.full((n, Tmax), -1, np.int64)
    for j in range(3):
        m = (t[None] >= bpos[:, j:j + 1]) & (t[None] < bpos[:, j:j + 1] + WIN)
        win[m] = j
    validt = torch.from_numpy(valid).to(device)
    wint = torch.from_numpy(win).to(device)
    bpt = torch.from_numpy(bpos).to(device)
    lab_flat = (wint >= 0)[validt]
    nvalid = int(valid.sum())
    res = {}
    for li in range(nL):
        X = torch.from_numpy(np.ascontiguousarray(sig[:, :, li].transpose(1, 2, 0, 3))).to(device)  # (C,H+1,n,T)
        X = X.reshape(nC * nH1, n, Tmax)
        Xv = X[:, validt]                                                     # (S, Nvalid)
        au = auroc(-Xv, lab_flat)
        m_in = Xv[:, ~lab_flat].mean(1).cpu().numpy()
        m_af = Xv[:, lab_flat].mean(1).cpu().numpy()
        ql = torch.tensor(QLEV, device=device, dtype=Xv.dtype)
        thr = torch.stack([torch.quantile(x, ql) for x in Xv])                  # (S,Q)
        del Xv
        Sg, Q = thr.shape
        Xn = torch.nan_to_num(X, nan=2.0)
        last = torch.full((Sg, Q, n), -10 ** 6, device=device, dtype=torch.long)
        prev = torch.ones((Sg, Q, n), device=device, dtype=torch.bool)
        fa = torch.zeros((Sg, Q), device=device)
        non = torch.zeros((Sg, Q), device=device)
        first = torch.full((Sg, Q, n, 3), -1, device=device, dtype=torch.long)
        for tt in range(WIN, Tmax):
            vt = validt[:, tt]
            below = (Xn[:, None, :, tt] < thr[:, :, None]) & vt
            onset = below & ~prev & (tt - last >= WIN)
            last = torch.where(onset, tt, last)
            prev = below | ~vt
            w = wint[:, tt]
            fa += (onset & (w < 0)).sum(-1)
            non += onset.sum(-1)
            if bool((w >= 0).any()):
                wc = w.clamp_min(0)
                cur = first.gather(-1, wc.view(1, 1, n, 1).expand(Sg, Q, n, 1)).squeeze(-1)
                setm = onset & (w >= 0) & (cur < 0)
                newv = torch.where(setm, tt - bpt.gather(1, wc[:, None]).squeeze(1), cur)
                first.scatter_(-1, wc.view(1, 1, n, 1).expand(Sg, Q, n, 1), newv[..., None])
        det = (first >= 0).float()                                           # (S,Q,n,3)
        rate = det.mean((-1, -2))
        tp = non - fa
        prec = tp / non.clamp_min(1)
        f1 = 2 * prec * rate / (prec + rate).clamp_min(1e-12)
        far = fa / nvalid * 1000
        bi = f1.argmax(1)
        ok = far <= 1.0
        bi1 = torch.where(ok.any(1), torch.where(ok, rate, -1.0).argmax(1), torch.zeros_like(bi))
        firstc, ratec, farc = first.cpu().numpy(), rate.cpu().numpy(), far.cpu().numpy()
        f1c, bic, bi1c, okc = f1.cpu().numpy(), bi.cpu().numpy(), bi1.cpu().numpy(), ok.any(1).cpu().numpy()
        thrc = thr.cpu().numpy()
        per = []
        for s in range(Sg):
            q = bic[s]
            fs = firstc[s, q]
            d = fs[fs >= 0]
            per.append({"auroc": float(au[s]), "mean_within": float(m_in[s]), "mean_after_boundary": float(m_af[s]),
                        "best_f1": {"f1": float(f1c[s, q]), "threshold": float(thrc[s, q]), "quantile": float(QLEV[q]),
                                    "detection_rate": float(ratec[s, q]), "fa_per_1000": float(farc[s, q]),
                                    "median_delay": float(np.median(d)) if len(d) else None,
                                    "detection_rate_by_boundary": [float((fs[:, j] >= 0).mean()) for j in range(3)]},
                        "best_at_fa_le_1": ({"detection_rate": float(ratec[s, bi1c[s]]),
                                             "fa_per_1000": float(farc[s, bi1c[s]]),
                                             "quantile": float(QLEV[bi1c[s]])} if okc[s] else None)})
        lay = {}
        for ci, (hs, hl) in enumerate(CFGS):
            ent = per[ci * nH1:(ci + 1) * nH1]
            heads, hm = ent[:H], ent[H]
            agg = lambda f: [f(e) for e in heads]
            lay[f"s{hs}_l{hl}"] = {
                "head_mean_signal": hm,
                "per_head_mean": {"auroc": float(np.mean(agg(lambda e: e["auroc"]))),
                                  "mean_within": float(np.mean(agg(lambda e: e["mean_within"]))),
                                  "mean_after_boundary": float(np.mean(agg(lambda e: e["mean_after_boundary"]))),
                                  "detection_rate": float(np.mean(agg(lambda e: e["best_f1"]["detection_rate"]))),
                                  "fa_per_1000": float(np.mean(agg(lambda e: e["best_f1"]["fa_per_1000"]))),
                                  "f1": float(np.mean(agg(lambda e: e["best_f1"]["f1"])))},
                "best_head": int(np.argmax(agg(lambda e: e["auroc"]))),
                "per_head": heads}
        res[f"layer_{li}"] = lay
        log(f"m1a layer {li} done")
    return res


# ================================================================ M1b span separation
def span_cos(A, B):
    """A, B (P, H, D) -> (P,) mean over heads of per-head cosine."""
    return cosd(A, B).mean(-1)


def m1b(kmean, vmean, sp, seed, device):
    n, nL = kmean.shape[:2]
    keys = akey(sp["src"], sp["doc"])                                      # (n,4)
    rng = np.random.default_rng([seed, 12])
    flat_src, flat_key = sp["src"].reshape(-1), keys.reshape(-1)
    fr = np.repeat(np.arange(n), 4)
    pairs_same_src, pairs_diff_src = [], []
    while len(pairs_same_src) < NPAIR or len(pairs_diff_src) < NPAIR:
        i, j = rng.integers(4 * n, size=2)
        if fr[i] == fr[j] or flat_key[i] == flat_key[j]:
            continue
        if flat_src[i] == flat_src[j]:
            if len(pairs_same_src) < NPAIR:
                pairs_same_src.append((i, j))
        elif len(pairs_diff_src) < NPAIR:
            pairs_diff_src.append((i, j))
    I0 = np.arange(n) * 4
    sets = {"same_doc_a1_a2": (I0, I0 + 2),
            "diff_doc_same_source": tuple(np.array(pairs_same_src).T),
            "diff_source": tuple(np.array(pairs_diff_src).T),
            "diff_source_a1_b_same_row": (I0, I0 + 1)}
    out = {}
    for name, feat in (("k", kmean), ("kv", np.concatenate([kmean, vmean], -1))):
        out[name] = {}
        for li in range(nL):
            X = torch.from_numpy(feat[:, li].reshape(n * 4, *feat.shape[3:])).to(device)
            ent = {}
            for sname, (i, j) in sets.items():
                c = span_cos(X[i], X[j]).cpu().numpy()
                ent[sname] = {"mean": float(c.mean()), "std": float(c.std()), "n": int(len(c))}
            ent["dprime_same_doc_vs_diff_doc_same_source"] = float(
                (ent["same_doc_a1_a2"]["mean"] - ent["diff_doc_same_source"]["mean"])
                / np.sqrt((ent["same_doc_a1_a2"]["std"] ** 2 + ent["diff_doc_same_source"]["std"] ** 2) / 2))
            out[name][f"layer_{li}"] = ent
    return out


# ================================================================ M1c selection (pass 2)
@torch.no_grad()
def m1c(cap, docs, sp, kmean, device):
    n, nL, _, H, D = kmean.shape
    nb = len(SEL_EDGES) - 1
    # acc[layer, query(raw,ema16), cand(2, 8), agg(per-head avg, head-mean score), bucket] sums and counts
    acc = np.zeros((nL, 2, 2, 2, nb + 1))
    cnt = np.zeros((nb + 1,))
    tot = sp["len"].sum(1)
    for idx in S.packs(list(tot + 1), BATCH_TOK):
        feats = cap([m1_row(docs, sp, r) for r in idx], device)
        for j, r in enumerate(idx):
            a = sp["len"][r, 0] + sp["len"][r, 1]
            L2 = sp["len"][r, 2]
            p = np.arange(L2)
            bk = np.searchsorted(SEL_EDGES, p, side="right") - 1
            cnt[:nb] += np.bincount(bk, minlength=nb)
            cnt[nb] += L2
            bkt = torch.from_numpy(bk).to(device)
            for li in range(nL):
                k = feats[li][j][0]
                cands = [kmean[r, li, 0], kmean[r, li, 1]] + [kmean[r2, li, s2] for r2, s2 in sp["dist"][r]]
                Cd = torch.from_numpy(np.stack(cands)).to(device)                 # (8,H,D)
                Cd = Cd / Cd.norm(dim=-1, keepdim=True)
                qs = [k[a:a + L2], ema(k, SEL_H)[a:a + L2].float()]
                for qi, qv in enumerate(qs):
                    qn = qv / qv.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    cs = torch.einsum("thd,chd->thc", qn, Cd)                    # (T2,H,8)
                    for ci, nc in enumerate((2, 2 + NDIST)):
                        c = cs[..., :nc]
                        ph = (c.argmax(-1) == 0).float().mean(-1)                # (T2,) per-head avg
                        hm = (c.mean(1).argmax(-1) == 0).float()                 # head-mean score
                        for ai, v in enumerate((ph, hm)):
                            sb = torch.zeros(nb, device=device).index_add_(0, bkt, v)
                            acc[li, qi, ci, ai, :nb] += sb.cpu().numpy()
                            acc[li, qi, ci, ai, nb] += float(v.sum())
        log(f"m1c rows {idx[-1] + 1}/{n}")
    accm = acc / cnt
    out = {}
    for li in range(nL):
        out[f"layer_{li}"] = {
            f"{q}_{c}_{a}": {"by_bucket": accm[li, qi, ci, ai, :nb].tolist(), "overall": float(accm[li, qi, ci, ai, nb])}
            for qi, q in enumerate(("rawk", f"ema{SEL_H}")) for ci, c in enumerate(("a1_vs_b", f"a1_b_plus{NDIST}"))
            for ai, a in enumerate(("perhead", "headmean"))}
    return out, cnt.tolist()


# ================================================================ main per run
@torch.no_grad()
def run_one(args):
    t0 = time.time()
    run = Path(args.run).resolve()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    eot = int(json.load(open(ROOT / "data/bin/meta.json"))["eot"])
    model, cfg, ckpt, step = S.load_model(run, device, args.ckpt)
    out = OUT / run.name
    out.mkdir(parents=True, exist_ok=True)
    log(f"run {run.name} ckpt {ckpt.name} step {step} gate_mode {model.c.gate_mode}")
    docs = S.load_docs(eot)
    head = {"run": run.name, "ckpt": ckpt.name, "step": step, "gate_mode": model.c.gate_mode, "seed": args.seed,
            "sources": SOURCES}
    warnings = []
    if args.part in ("all", "m3"):
        a_src, a_doc, b_src, b_doc = build_m3(docs, args.rows3, args.seed)
        tl = S.TokenLoss(model, device)
        vals = run_m3(tl, docs, a_src, a_doc, b_src, b_doc)
        tl.close()
        if tl.max_dev > 2e-2:
            warnings.append(f"per-token loss deviates from fused loss by up to {tl.max_dev:.4f}")
        h = S.spec_hash(a_src, a_doc, b_src, b_doc)
        np.savez(out / "m3_rows.npz", spec_hash=h, a_src=a_src, a_doc=a_doc, b_src=b_src, b_doc=b_doc, **vals)
        E.dump(out, "m3_preserve.json", {
            **head, "rows": len(a_src), "spec_hash": h, "B_lengths": list(M3_BL), "bucket_edges": M3_EDGES,
            "bucket_note": "columns: buckets [0,16),[16,64),[64,256),[256,511) of A2 input positions, then overall",
            "conditions": {"i_L": "A1|B[:L]|A2", "ii": "A1|A2 (oracle)", "iii": "A2 alone (cold)", "iv_L": "B[:L]|A2"},
            "unique_A_docs": S.unique_docs(a_src, a_doc), "unique_B_docs": S.unique_docs(b_src, b_doc),
            "max_forward_dev_tokenwise_vs_fused": tl.max_dev, "stats": m3_stats(vals, a_src, a_doc),
            "warnings": list(warnings), "seconds": time.time() - t0}, warnings)
        log(f"m3 written {time.time() - t0:.0f}s")
    if args.part in ("all", "m1"):
        cap = Capture(model)
        m1 = {**head, "rows_per_variant": args.rows1, "cfgs": [f"s{a}_l{b}" for a, b in CFGS], "window": WIN,
              "detector": ("alarm at t if signal_t < thr, signal_{t-1} >= thr and no alarm in the previous 32 tokens; "
                           "tokens t < 32 of a row excluded; an alarm in [b, b+32) of a true boundary b detects it, "
                           "any other alarm is a false alarm; thresholds = quantiles " + str(list(np.round(QLEV, 5)))
                           + " of the signal; best = max F1 (precision = in-window alarms / alarms, recall = "
                           "detection rate); fa_per_1000 per 1000 valid tokens"),
              "variants": {}}
        m2 = {**head, "span_lengths": list(M2_L), "m": list(M2_M),
              "definition": "M = sum_i beta_i v_i k_i^T over the first L tokens of each of the 4 segments (fixed "
                            "variant); fraction of ||M||_F^2 in the top-m singular values; mean over rows, segments "
                            "and heads; participation ratio = (sum s^2)^2 / sum s^4", "layers": {}}
        for variant in ("fixed", "variable"):
            tv = time.time()
            sp = build_m1(docs, args.rows1, args.seed, variant == "variable")
            do_m2 = variant == "fixed"
            sig, kmean, vmean, fr, pr = pass1(cap, docs, sp, device, do_m2)
            log(f"{variant} pass1 {time.time() - tv:.0f}s")
            a = m1a(sig, sp, device)
            del sig
            b = m1b(kmean, vmean, sp, args.seed, device)
            c, ccnt = m1c(cap, docs, sp, kmean, device)
            m1["variants"][variant] = {"segment_lengths_mean": sp["len"].mean(0).tolist(),
                                       "boundary_detection": a, "span_separation": b,
                                       "selection": {"bucket_edges": SEL_EDGES, "token_counts": ccnt, "layers": c}}
            if do_m2:
                for li in range(fr.shape[1]):
                    m2["layers"][f"layer_{li}"] = {
                        f"L{L}": {"frac_top_m_mean": fr[:, li, :, Li].mean((0, 1, 2)).tolist(),
                                  "frac_top_m_std": fr[:, li, :, Li].reshape(-1, len(M2_M)).std(0).tolist(),
                                  "participation_ratio_mean": float(pr[:, li, :, Li].mean())}
                        for Li, L in enumerate(M2_L)}
                np.savez(out / "m2_frac.npz", frac=fr, pr=pr)
            log(f"{variant} done {time.time() - tv:.0f}s")
        E.dump(out, "m1_pooling.json", m1, warnings)
        E.dump(out, "m2_rank.json", m2, warnings)
    if args.part == "m2":
        tv = time.time()
        cap = Capture(model)
        sp = build_m1(docs, args.rows1, args.seed, False)                         # same rows as the fixed variant
        fr, pr, frc, prc, shared, resid = run_m2(cap, docs, sp, device)
        m2 = {**head, "rows": args.rows1, "span_lengths": list(M2_L), "m": list(M2_M),
              "definition": "M = sum_i beta_i v_i k_i^T over the first L tokens of each of the 4 segments (fixed "
                            "variant); fraction of ||M||_F^2 in the top-m singular values; mean over rows, segments "
                            "and heads; participation ratio = (sum s^2)^2 / sum s^4",
              "centred_definition": "per layer and head, Mbar = mean of M/L over all spans (rows x 4 segments x "
                                    "L in span_lengths); M_c = M - L*Mbar; same statistics on M_c. "
                                    "shared_norm_share = ||L*Mbar||_F^2 / ||M||_F^2, centred_norm_share = "
                                    "||M_c||_F^2 / ||M||_F^2 (mean over rows, segments, heads)",
              "layers": {}, "layers_centred": {}}
        for li in range(fr.shape[1]):
            m2["layers"][f"layer_{li}"] = {
                f"L{L}": {"frac_top_m_mean": fr[:, li, :, Li].mean((0, 1, 2)).tolist(),
                          "frac_top_m_std": fr[:, li, :, Li].reshape(-1, len(M2_M)).std(0).tolist(),
                          "participation_ratio_mean": float(pr[:, li, :, Li].mean())}
                for Li, L in enumerate(M2_L)}
            m2["layers_centred"][f"layer_{li}"] = {
                f"L{L}": {"frac_top_m_mean": frc[:, li, :, Li].mean((0, 1, 2)).tolist(),
                          "frac_top_m_std": frc[:, li, :, Li].reshape(-1, len(M2_M)).std(0).tolist(),
                          "participation_ratio_mean": float(prc[:, li, :, Li].mean()),
                          "shared_norm_share_mean": float(shared[:, li, :, Li].mean()),
                          "centred_norm_share_mean": float(resid[:, li, :, Li].mean())}
                for Li, L in enumerate(M2_L)}
        np.savez(out / "m2_frac.npz", frac=fr, pr=pr, frac_centred=frc, pr_centred=prc, shared_norm_share=shared,
                 centred_norm_share=resid)
        E.dump(out, "m2_rank.json", m2, warnings)
        log(f"m2 done {time.time() - tv:.0f}s")
    for w in warnings:
        log(f"WARNING: {w}")
    log(f"done {out} {time.time() - t0:.0f}s")


# ================================================================ summary
def f3(x, se=None):
    return f"{x:+.3f}±{se:.3f}" if se is not None else f"{x:.3f}"


def summarize(args):
    runs = [Path(p).resolve().name for p in args.summarize]
    lines = ["# Verification of attention-redesign premises (training-free)", "",
             f"Runs: A = `{runs[0]}`, B = `{runs[1]}`. No state reset inside any row. Details/definitions: "
             "`Eval/verify_structure.py` docstring and the json files in `results/verify/<run>/`.", ""]
    # ---- M3
    lines += ["## M3: loss on A2 (nats/token), upper bound of gain from preserving A's state", "",
              "Columns: A2 input-position buckets [0,16) [16,64) [64,256) [256,511) and overall. ± = SE with rows as "
              "unit (se_doc, A-document clusters, in json). (i) A1|B|A2, (ii) A1|A2, (iii) A2, (iv) B|A2.", ""]
    z = []
    for rn in runs:
        p = OUT / rn / "m3_preserve.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        st = d["stats"]
        lines += [f"### {rn} (rows {d['rows']})", "", "| quantity | 0-15 | 16-63 | 64-255 | 256-510 | overall |",
                  "|---|---|---|---|---|---|"]
        for c in ["ii", "iii"] + [f"i_{L}" for L in M3_BL] + [f"iv_{L}" for L in M3_BL]:
            e = st["levels"][c]
            lines.append(f"| level {c} | " + " | ".join(f3(v) for v in e["mean"]) + " |")
        for c in ["iii_minus_ii"] + [f"i_minus_ii_L{L}" for L in M3_BL] + [f"iv_minus_ii_L{L}" for L in M3_BL]:
            e = st["diffs"][c]
            lines.append(f"| {c} | " + " | ".join(f3(v, s) for v, s in zip(e["mean"], e["se_row"])) + " |")
        lines.append("")
        z.append(np.load(OUT / rn / "m3_rows.npz"))
    if len(z) == 2 and str(z[0]["spec_hash"]) == str(z[1]["spec_hash"]):
        rows = np.arange(len(z[0]["ii"]))
        keys = akey(z[0]["a_src"], z[0]["a_doc"])
        pd = {}
        for c in ["iii"] + [f"i_{L}" for L in M3_BL]:
            nm = "iii_minus_ii" if c == "iii" else f"i_minus_ii_L{c[2:]}"
            pd[nm] = m3_entry((z[1][c] - z[1]["ii"]) - (z[0][c] - z[0]["ii"]), rows, keys)
        pd["ii"] = m3_entry(z[1]["ii"] - z[0]["ii"], rows, keys)
        E.dump(OUT, "paired_m3.json", {"A": runs[0], "B": runs[1], "difference": "B - A, paired by row",
                                       "stats": pd}, [])
        lines += [f"### paired B - A (same rows)", "", "| quantity | 0-15 | 16-63 | 64-255 | 256-510 | overall |",
                  "|---|---|---|---|---|---|"]
        for nm, e in pd.items():
            lines.append(f"| {nm} | " + " | ".join(f3(v, s) for v, s in zip(e["mean"], e["se_row"])) + " |")
        lines.append("")
    # ---- M1 / M2
    for rn in runs:
        p = OUT / rn / "m1_pooling.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        lines += [f"## M1 ({rn})", ""]
        for vname, v in d["variants"].items():
            bd = v["boundary_detection"]
            nL = len(bd)
            lines += [f"### M1a boundary detection, {vname} segments (rows {d['rows_per_variant']})", "",
                      "Signal = cos(EMA_short(k), EMA_long(k)). Per layer, config with the best head-mean-signal "
                      "AUROC. hm = head-mean signal (mean of per-head cosines); ph = per-head metric averaged over "
                      "heads; max = best head. Det/FA/delay for hm at the max-F1 threshold; det@FA<=1 = best detection "
                      "rate with <= 1 false alarm per 1000 tokens.", "",
                      "| layer | cfg | AUROC hm | AUROC ph | AUROC max | sig within | sig after bnd | det | FA/1k | "
                      "delay | det@FA<=1 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for li in range(nL):
                lay = bd[f"layer_{li}"]
                cbest = max(lay, key=lambda c: lay[c]["head_mean_signal"]["auroc"])
                e = lay[cbest]
                hm = e["head_mean_signal"]
                bf = hm["best_f1"]
                amax = max(h["auroc"] for h in e["per_head"])
                d1 = hm["best_at_fa_le_1"]
                lines.append(f"| {li} | {cbest} | {hm['auroc']:.3f} | {e['per_head_mean']['auroc']:.3f} | {amax:.3f} | "
                             f"{hm['mean_within']:.3f} | {hm['mean_after_boundary']:.3f} | "
                             f"{bf['detection_rate']:.2f} | {bf['fa_per_1000']:.2f} | {bf['median_delay']} | "
                             f"{(d1['detection_rate'] if d1 else float('nan')):.2f} |")
            lines.append("")
            lines += [f"AUROC (head-mean signal) for every config, mean over layers: " + ", ".join(
                f"{c} {np.mean([bd[f'layer_{li}'][c]['head_mean_signal']['auroc'] for li in range(nL)]):.3f}"
                for c in d["cfgs"]), ""]
            ss = v["span_separation"]
            lines += [f"### M1b span-summary cosine, {vname} (mean over heads of per-head cosine; mean ± std over pairs)",
                      "", "| layer | feat | same doc a1-a2 | diff doc same src | diff src | a1-b same row | d' |",
                      "|---|---|---|---|---|---|---|"]
            for li in range(nL):
                for ft in ("k", "kv"):
                    e = ss[ft][f"layer_{li}"]
                    g = lambda s: f"{e[s]['mean']:.3f}±{e[s]['std']:.3f}"
                    lines.append(f"| {li} | {ft} | {g('same_doc_a1_a2')} | {g('diff_doc_same_source')} | "
                                 f"{g('diff_source')} | {g('diff_source_a1_b_same_row')} | "
                                 f"{e['dprime_same_doc_vs_diff_doc_same_source']:.2f} |")
            lines.append("")
            sel = v["selection"]
            lines += [f"### M1c selection top-1 accuracy (a1 chosen), {vname}", "",
                      "Query = raw k or EMA16(k) at A2 tokens; candidates = span means of k. perhead = per-head "
                      "argmax accuracy averaged over heads; headmean = argmax of head-averaged cosine. Chance: 0.5 "
                      f"(2 cands), 0.125 (8 cands). Position buckets {sel['bucket_edges']}.", "",
                      "| layer | raw 2 ph | raw 2 hm | ema 2 ph | ema 2 hm | raw 8 ph | raw 8 hm | ema 8 ph | ema 8 hm "
                      "| ema 8 hm by pos |", "|---|---|---|---|---|---|---|---|---|---|"]
            for li in range(nL):
                e = sel["layers"][f"layer_{li}"]
                cols = []
                for nc in ("a1_vs_b", f"a1_b_plus{NDIST}"):
                    for q in ("rawk", f"ema{SEL_H}"):
                        for a in ("perhead", "headmean"):
                            cols.append(f"{e[f'{q}_{nc}_{a}']['overall']:.3f}")
                bp = " / ".join(f"{x:.2f}" for x in e[f"ema{SEL_H}_a1_b_plus{NDIST}_headmean"]["by_bucket"])
                lines.append(f"| {li} | " + " | ".join(cols) + f" | {bp} |")
            lines.append("")
        p2 = OUT / rn / "m2_rank.json"
        if p2.exists():
            d2 = json.load(open(p2))
            lines += [f"## M2 ({rn}): fraction of ||sum beta v k^T||_F^2 in top-m singular values", "",
                      "Mean over rows x 4 segments x heads. PR = participation ratio (effective rank).", "",
                      "| layer | L | " + " | ".join(f"m={m}" for m in d2["m"]) + " | PR |",
                      "|---|---|" + "---|" * len(d2["m"]) + "---|"]
            for ln, e in d2["layers"].items():
                for L in d2["span_lengths"]:
                    x = e[f"L{L}"]
                    lines.append(f"| {ln.split('_')[1]} | {L} | " + " | ".join(f"{v:.3f}" for v in x["frac_top_m_mean"])
                                 + f" | {x['participation_ratio_mean']:.1f} |")
            lines.append("")
            if "layers_centred" in d2:
                lines += [f"## M2 centred ({rn}): same, after subtracting the shared component L*mean(M/L)", "",
                          "Per layer and head, Mbar = mean of M/L over all spans (rows x 4 segments x L); "
                          "M_c = M - L*Mbar. shared = ||L*Mbar||_F^2 / ||M||_F^2, resid = ||M_c||_F^2 / ||M||_F^2 "
                          "(means over rows x 4 segments x heads).", "",
                          "| layer | L | " + " | ".join(f"m={m}" for m in d2["m"]) + " | PR | shared | resid |",
                          "|---|---|" + "---|" * len(d2["m"]) + "---|---|---|"]
                for ln, e in d2["layers_centred"].items():
                    for L in d2["span_lengths"]:
                        x = e[f"L{L}"]
                        lines.append(f"| {ln.split('_')[1]} | {L} | "
                                     + " | ".join(f"{v:.3f}" for v in x["frac_top_m_mean"])
                                     + f" | {x['participation_ratio_mean']:.1f} | {x['shared_norm_share_mean']:.3f} | "
                                     f"{x['centred_norm_share_mean']:.3f} |")
                lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {OUT / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--summarize", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--part", default="all", choices=("all", "m3", "m1", "m2"))
    ap.add_argument("--rows3", type=int, default=1000)
    ap.add_argument("--rows1", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out:
        globals()["OUT"] = Path(args.out).resolve()
    if bool(args.run) == bool(args.summarize):
        ap.error("give exactly one of --run / --summarize")
    try:
        summarize(args) if args.summarize else run_one(args)
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        E.die(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
