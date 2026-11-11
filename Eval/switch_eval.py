"""Context switches without state reset: how a model behaves when one row holds several unrelated contexts.

Usage (from the research-llm directory):
    .venv/bin/python Eval/switch_eval.py --run model/<run dir> --device cuda:<N> [--rows 2000] [--seed 0]
        -> results/switch/<run name>/{exp1_switch,exp2_interference}.json + rows.npz (per-row data for pairing)
    .venv/bin/python Eval/switch_eval.py --compare model/<run A> model/<run B>
        -> results/switch/paired_<run B>_minus_<run A>/{exp1_switch,exp2_interference}.json  (CPU only, B - A by row)

Documents: val.bin of each source = whole documents, each followed by eot. A "segment" is a slice of one document
starting at its document start (Exp 1: tokens [0,512); Exp 2: A1 = [0,512), A2 = [512,1024), B = [0,L)). No eot is
inserted between segments. Rows are packed into one flattened sequence per forward (cu_seqlens at row starts, so
nothing carries between rows); "reset" adds cu_seqlens at segment starts (conv, DeltaNet state and pool all reset,
as for a document start in training).

Exp 1 (switch): rows A|B|C|D of 4 x 512 tokens, segment j of row r from source (r + j) mod 5 (consecutive segments
always from different sources), docs with >= 512 tokens drawn without replacement from a shuffled per-source list
(reshuffled when exhausted). (a) reset at each segment start, (b) no reset.
Exp 2 (interference): A >= 1024 tokens, A source = r mod 5; B from another source (cycled), B >= 1024 tokens, same B
doc for every L. (i) A1|A2, (ii) A1|B[:L]|A2, (iii) A2 alone; loss on A2 only.

Position buckets: a prediction belongs to the segment of its input token, at position p within that segment; its
target is the next token of the same segment, so the last position of every segment (whose target is the next
segment's first token, or not in the row) is excluded: p in [0, 511), bucket [256,512) holds p = 256..510.
"overall" = token-weighted mean over all buckets (weights = bucket token counts).

Standard errors: se_row = SE of the mean with rows as the unit (for pooled segments: cluster-robust over rows);
se_doc = cluster-robust SE with the scored segment's document (Exp 1) or document A (Exp 2) as cluster, because
documents are reused across rows (val data has few long documents, gutenberg only 17). Retained fraction
((iii)-(ii))/((iii)-(i)) is a ratio of means; its SEs are bootstrap (rows / A-doc clusters resampled).
"""
import argparse
import hashlib
import json
import math
import sys
import time
import tomllib
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Eval"))
import evaluate as E  # noqa: E402  (sets HIPBLASLT path, imports Train/model.py as E.M)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

M = E.M
SOURCES = ["dclm", "fineweb_edu", "wikipedia", "gutenberg", "cosmopedia"]
SEG = 512
NSEG = 4
LS = (64, 256, 1024)
EDGES = [0] + [2 ** i for i in range(int(math.log2(SEG)))] + [SEG]   # [0,1,2,4,...,256,512]
NB = len(EDGES) - 1
P = np.arange(SEG - 1)                                              # scored positions within a segment
BIDX = np.searchsorted(EDGES, P, side="right") - 1
CNT = np.bincount(BIDX, minlength=NB).astype(np.float64)            # [1,1,2,4,...,128,255]
WB = CNT / CNT.sum()
TOK_BUDGET = 16384     # input tokens per forward
NBOOT = 2000
SEG0_TOL = 1e-3        # Exp 1 check: |mean (b)-(a)| on segment 0, any bucket


def log(msg):
    print(f"[switch] {msg}", flush=True)


# ---------------------------------------------------------------- data
def load_docs(eot):
    """{source: (memmap, doc start offsets, doc lengths without eot)}"""
    docs = {}
    for s in SOURCES:
        m = np.memmap(ROOT / "data/bin" / s / "val.bin", dtype=np.uint16, mode="r")
        e = np.flatnonzero(m == eot)
        st = np.concatenate([[0], e[:-1] + 1]).astype(np.int64)
        docs[s] = (m, st, e - st)
    return docs


def toks(docs, s, d, a, b):
    m, st, ln = docs[SOURCES[s]]
    assert b <= ln[d], (SOURCES[s], d, b, ln[d])
    return np.asarray(m[st[d] + a:st[d] + b], dtype=np.int64)


class Stream:
    """Shuffled list of doc indices, drawn without replacement and reshuffled when exhausted."""

    def __init__(self, ids, rng):
        self.ids, self.rng, self.buf = np.asarray(ids), rng, []

    def next(self):
        if not self.buf:
            self.buf = list(self.rng.permutation(self.ids)[::-1])
        return int(self.buf.pop())


def build_exp1(docs, n, seed):
    rng = np.random.default_rng([seed, 1])
    streams = [Stream(np.flatnonzero(docs[s][2] >= SEG), rng) for s in SOURCES]
    src = (np.arange(n)[:, None] + np.arange(NSEG)[None, :]) % len(SOURCES)
    doc = np.array([[streams[k].next() for k in row] for row in src], dtype=np.int64)
    return src, doc


def build_exp2(docs, n, seed):
    rng = np.random.default_rng([seed, 2])
    ok = [np.flatnonzero(docs[s][2] >= 2 * SEG) for s in SOURCES]
    sa = [Stream(ids, rng) for ids in ok]
    sb = [Stream(ids, rng) for ids in ok]
    r = np.arange(n)
    a_src = r % len(SOURCES)
    b_src = (a_src + 1 + (r // len(SOURCES)) % (len(SOURCES) - 1)) % len(SOURCES)
    a_doc = np.array([sa[k].next() for k in a_src], dtype=np.int64)
    b_doc = np.array([sb[k].next() for k in b_src], dtype=np.int64)
    assert max(LS) <= 2 * SEG
    return a_src, a_doc, b_src, b_doc


def spec_hash(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- model
def load_model(run, device, ckpt_arg=None):
    cfg = tomllib.load(open(run / "config.toml", "rb"))
    if not cfg["data"].get("doc_mask", True):
        E.die("config has doc_mask = false")
    ckpt = E.pick_ckpt(run, ckpt_arg)
    model = M.LM(M.ModelConfig(**cfg["model"])).to(device).eval()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    step = ck.get("step")
    del ck
    return model, cfg, ckpt, step


class TokenLoss:
    """Per-token CE of packed rows (rows independent, optional resets inside rows), as in Eval/evaluate.py:
    hidden state from a hook on model.norm, bf16 LM head, fused loss of LM.forward kept for a consistency check."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.hidden = {}
        self.h = model.norm.register_forward_hook(lambda mod, i, o: self.hidden.__setitem__("h", o))
        self.W = model.embed.weight.to(torch.bfloat16)
        self.max_dev = 0.0

    def close(self):
        self.h.remove()

    @torch.no_grad()
    def __call__(self, rows, resets):
        """rows: list of 1-D int64 token arrays; resets: list of lists of in-row offsets where state resets.
        Returns list of arrays tok[i] (len(rows[i]) - 1): CE of predicting rows[i][p+1] from input position p."""
        xs, ys, cuts, off = [], [], [0], 0
        for r, rs in zip(rows, resets):
            n = len(r) - 1
            xs.append(r[:-1])
            ys.append(r[1:])
            cuts += [off + c for c in rs if 0 < c < n]
            off += n
            cuts.append(off)
        cu_cpu = torch.tensor(sorted(set(cuts)), dtype=torch.int64)
        x = torch.from_numpy(np.concatenate(xs)).to(self.device)[None]
        y = torch.from_numpy(np.concatenate(ys)).to(self.device)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = self.model(x, y, cu_cpu.to(self.device), cu_cpu)
        h = self.hidden.pop("h")[0].to(torch.bfloat16)
        yf = y.reshape(-1)
        tok = torch.empty(yf.numel(), dtype=torch.float32, device=self.device)
        for c0 in range(0, yf.numel(), E.LOGIT_CHUNK):
            lgt = F.linear(h[c0:c0 + E.LOGIT_CHUNK], self.W).float()
            tok[c0:c0 + E.LOGIT_CHUNK] = F.cross_entropy(lgt, yf[c0:c0 + E.LOGIT_CHUNK], reduction="none")
        self.max_dev = max(self.max_dev, abs(float(tok.double().mean()) - float(loss.float())))
        tok = tok.cpu().numpy().astype(np.float64)
        return np.split(tok, np.cumsum([len(r) - 1 for r in rows])[:-1])


def bucket(tok, o):
    """Bucket means of the scored predictions of the segment starting at input offset o."""
    return np.bincount(BIDX, weights=tok[o:o + SEG - 1], minlength=NB) / CNT


def packs(lengths, budget=TOK_BUDGET):
    out, cur, n = [], [], 0
    for i, ln in enumerate(lengths):
        if cur and n + ln - 1 > budget:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += ln - 1
    if cur:
        out.append(cur)
    return out


def run_exp1(tl, docs, src, doc):
    R = len(src)
    res = {c: np.zeros((R, NSEG, NB)) for c in ("a", "b")}
    seg0_tok = 0.0
    for idx in packs([NSEG * SEG] * R):
        rows = [np.concatenate([toks(docs, src[r, j], doc[r, j], 0, SEG) for j in range(NSEG)]) for r in idx]
        out = {}
        for c, rs in (("a", [j * SEG for j in range(1, NSEG)]), ("b", [])):
            out[c] = tl(rows, [rs] * len(rows))
            for k, r in enumerate(idx):
                for j in range(NSEG):
                    res[c][r, j] = bucket(out[c][k], j * SEG)
        seg0_tok = max(seg0_tok, max(float(np.abs(ta[:SEG - 1] - tb[:SEG - 1]).max()) for ta, tb in zip(out["a"], out["b"])))
    return res, seg0_tok


def run_exp2(tl, docs, a_src, a_doc, b_src, b_doc):
    R = len(a_src)
    conds = ["i", "iii"] + [f"ii_{L}" for L in LS]
    res = {c: np.zeros((R, NB)) for c in conds}
    A1 = lambda r: toks(docs, a_src[r], a_doc[r], 0, SEG)
    A2 = lambda r: toks(docs, a_src[r], a_doc[r], SEG, 2 * SEG)
    mk = {"i": lambda r: (np.concatenate([A1(r), A2(r)]), SEG),
          "iii": lambda r: (A2(r), 0)}
    for L in LS:
        mk[f"ii_{L}"] = (lambda L: lambda r: (np.concatenate([A1(r), toks(docs, b_src[r], b_doc[r], 0, L), A2(r)]),
                                             SEG + L))(L)
    for c in conds:
        built = [mk[c](r) for r in range(R)]
        for idx in packs([len(b[0]) for b in built]):
            out = tl([built[r][0] for r in idx], [[] for _ in idx])
            for k, r in enumerate(idx):
                res[c][r] = bucket(out[k], built[r][1])
        log(f"exp2 {c} done")
    return res


# ---------------------------------------------------------------- statistics
def cluster_se(x, g):
    """x (n, k) observations, g (n,) cluster ids: cluster-robust SE of the column means (G/(G-1) correction).
    With one observation per cluster this is the usual sample std / sqrt(n)."""
    n = x.shape[0]
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    if G < 2:
        return np.full(x.shape[1:], np.nan)
    S = np.zeros((G,) + x.shape[1:])
    np.add.at(S, inv, x - x.mean(0))
    return np.sqrt(G / (G - 1) * (S ** 2).sum(0)) / n


def summ(x, rows, docs):
    """x (n, NB) bucket means per observation -> mean, se_row, se_doc per bucket and overall."""
    X = np.column_stack([x, x @ WB])
    m, sr, sd = X.mean(0), cluster_se(X, rows), cluster_se(X, docs)
    return {"n_obs": int(len(x)), "n_rows": int(len(np.unique(rows))), "n_docs": int(len(np.unique(docs))),
            "loss": m[:NB].tolist(), "se_row": sr[:NB].tolist(), "se_doc": sd[:NB].tolist(),
            "overall": float(m[NB]), "overall_se_row": float(sr[NB]), "overall_se_doc": float(sd[NB])}


def exp1_groups(src):
    """(name, mask over (row, seg) observations)."""
    R = len(src)
    seg = np.broadcast_to(np.arange(NSEG), (R, NSEG))
    g = [(f"segment_{j}", seg == j) for j in range(NSEG)]
    g.append(("segments_1_3", seg >= 1))
    g += [(f"source_segments_1_3:{s}", (seg >= 1) & (src == k)) for k, s in enumerate(SOURCES)]
    g += [(f"source_all_segments:{s}", src == k) for k, s in enumerate(SOURCES)]
    return g


def exp1_stats(vals, src, doc):
    """vals: {cond: (R, NSEG, NB)}; returns nested dict group -> cond -> summ."""
    R = len(src)
    rows = np.broadcast_to(np.arange(R)[:, None], (R, NSEG))
    dkey = src * 1_000_000 + doc
    out = {}
    for name, msk in exp1_groups(src):
        grp, _, sub = name.partition(":")
        d = {c: summ(v[msk], rows[msk], dkey[msk]) for c, v in vals.items()}
        if sub:
            out.setdefault(grp, {})[sub] = d
        else:
            out[grp] = d
    return out


def boot_weights(ids, rng, nboot=NBOOT):
    _, inv = np.unique(ids, return_inverse=True)
    G = inv.max() + 1
    return rng.multinomial(G, np.full(G, 1.0 / G), size=nboot)[:, inv].astype(np.float64)


def retained(i, ii, iii, W=None):
    """Ratio of means ((iii)-(ii))/((iii)-(i)) per bucket + overall; W (nboot, n) bootstrap weights or None."""
    cols = lambda x: np.column_stack([x, x @ WB])
    if W is None:
        mi, mii, miii = (cols(x).mean(0) for x in (i, ii, iii))
    else:
        mi, mii, miii = ((W @ cols(x)) / W.sum(1, keepdims=True) for x in (i, ii, iii))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (miii - mii) / (miii - mi)


def ratio_entry(r, se_row, se_doc):
    return {"value": r[:NB].tolist(), "se_row_boot": se_row[:NB].tolist(), "se_doc_boot": se_doc[:NB].tolist(),
            "overall": float(r[NB]), "overall_se_row_boot": float(se_row[NB]), "overall_se_doc_boot": float(se_doc[NB])}


def exp2_stats(models, a_src, a_doc, seed):
    """models: list of {cond: (R, NB)} (one model, or [A, B] for the paired B - A). Returns group -> L -> entries."""
    R = len(a_src)
    akey = a_src * 1_000_000 + a_doc
    rows = np.arange(R)
    out = {}
    groups = [("all", np.ones(R, bool))] + [(s, a_src == k) for k, s in enumerate(SOURCES)]
    for gi, (name, msk) in enumerate(groups):
        rng = np.random.default_rng([seed, 3, gi])
        Wr, Wd = boot_weights(rows[msk], rng), boot_weights(akey[msk], rng)
        sel = [{c: v[msk] for c, v in m.items()} for m in models]
        lev = lambda c: (sel[0][c] if len(sel) == 1 else sel[1][c] - sel[0][c])
        g = {"i": summ(lev("i"), rows[msk], akey[msk]), "iii": summ(lev("iii"), rows[msk], akey[msk]),
             "iii_minus_i": summ(lev("iii") - lev("i"), rows[msk], akey[msk])}
        for L in LS:
            c = f"ii_{L}"
            rr = [retained(s["i"], s[c], s["iii"]) for s in sel]
            br = [retained(s["i"], s[c], s["iii"], Wr) for s in sel]
            bd = [retained(s["i"], s[c], s["iii"], Wd) for s in sel]
            if len(sel) == 1:
                r, sr, sd = rr[0], br[0], bd[0]
            else:
                r, sr, sd = rr[1] - rr[0], br[1] - br[0], bd[1] - bd[0]
            g[f"L_{L}"] = {"ii": summ(lev(c), rows[msk], akey[msk]),
                           "ii_minus_i": summ(lev(c) - lev("i"), rows[msk], akey[msk]),
                           "iii_minus_ii": summ(lev("iii") - lev(c), rows[msk], akey[msk]),
                           "retained_fraction": ratio_entry(r, np.nanstd(sr, 0, ddof=1), np.nanstd(sd, 0, ddof=1))}
        out[name] = g
    return out


# ---------------------------------------------------------------- printing
def fmt(v, se=None, w=7, p=3):
    if v is None or not np.isfinite(v):
        return " " * (w - 3) + "nan"
    s = f"{v:+{w}.{p}f}" if se is not None else f"{v:{w}.{p}f}"
    return s + (f"±{se:.{p}f}" if se is not None else "")


def print_exp1(st, title):
    log(f"{title}: (b)-(a) by bucket {EDGES}")
    for name in ["segment_0", "segment_1", "segment_2", "segment_3", "segments_1_3"]:
        e = st[name]["b_minus_a"]
        log(f"  {name:28s} " + " ".join(fmt(v, s) for v, s in zip(e["loss"], e["se_row"]))
            + f" | overall {fmt(e['overall'], e['overall_se_row'])} (se_doc {e['overall_se_doc']:.3f})")
    for s in SOURCES:
        e = st["source_segments_1_3"][s]["b_minus_a"]
        log(f"  seg1-3 {s:21s} " + " ".join(fmt(v, x) for v, x in zip(e["loss"], e["se_row"]))
            + f" | overall {fmt(e['overall'], e['overall_se_row'])} (se_doc {e['overall_se_doc']:.3f})")


def print_exp2(st, title):
    log(f"{title}: A2 loss overall (se_row), retained fraction overall (se_row_boot / se_doc_boot)")
    for g, d in st.items():
        line = f"  {g:12s} i {fmt(d['i']['overall'], d['i']['overall_se_row'])} iii {fmt(d['iii']['overall'], d['iii']['overall_se_row'])}"
        for L in LS:
            e = d[f"L_{L}"]
            rf = e["retained_fraction"]
            line += (f" | L{L}: ii {fmt(e['ii']['overall'], e['ii']['overall_se_row'])} "
                     f"R {fmt(rf['overall'], rf['overall_se_row_boot'])}/{rf['overall_se_doc_boot']:.3f}")
        log(line)


# ---------------------------------------------------------------- main
def unique_docs(src, doc):
    return {s: int(len(np.unique(doc[src == k]))) for k, s in enumerate(SOURCES)}


DEF_EXP1 = ("Rows of 4 segments x 512 tokens; segment j of row r = first 512 tokens of a val document of source "
            "(r+j) mod 5 (docs >= 512 tokens). (a) reset: cu_seqlens at each segment start; (b) no reset: one sequence "
            "per row. loss[bucket] = mean over observations of the per-observation bucket mean; an observation is one "
            "(row, segment). Scored positions p in [0,511) of each segment (target inside the segment).")
DEF_EXP2 = ("A (>= 1024 tokens, source r mod 5): A1 = [0,512), A2 = [512,1024); B = first L tokens of a doc (>= 1024 "
            "tokens) of another source, same B for every L. (i) A1|A2, (ii) A1|B|A2, (iii) A2 alone; no reset inside "
            "any row. Loss on A2 positions p in [0,511). retained_fraction = ((iii)-(ii))/((iii)-(i)) of mean losses.")


@torch.no_grad()
def run_one(args):
    t0 = time.time()
    run = Path(args.run).resolve()
    device = torch.device(args.device)
    if device.type != "cuda":
        E.die("--device must be cuda:N")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    eot = int(json.load(open(ROOT / "data/bin/meta.json"))["eot"])
    model, cfg, ckpt, step = load_model(run, device, args.ckpt)
    out = Path(args.out).resolve() / run.name
    out.mkdir(parents=True, exist_ok=True)
    log(f"run {run.name} ckpt {ckpt.name} step {step} gate_mode {model.c.gate_mode} device {device}")
    docs = load_docs(eot)
    src, doc = build_exp1(docs, args.rows, args.seed)
    a_src, a_doc, b_src, b_doc = build_exp2(docs, args.rows, args.seed)
    h1, h2 = spec_hash(src, doc), spec_hash(a_src, a_doc, b_src, b_doc)
    warnings = []
    tl = TokenLoss(model, device)

    v1, seg0_tok = run_exp1(tl, docs, src, doc)
    v1["b_minus_a"] = v1["b"] - v1["a"]
    seg0 = v1["b_minus_a"][:, 0]
    seg0_max = float(np.abs(seg0.mean(0)).max())
    seg0_ok = seg0_max < SEG0_TOL
    log(f"exp1 done; segment 0 check: max |mean (b)-(a)| over buckets {seg0_max:.2e}, max per-token |diff| "
        f"{seg0_tok:.2e} -> {'OK' if seg0_ok else 'FAIL'}")
    if not seg0_ok:
        warnings.append(f"segment 0 (b)-(a) not ~0: max bucket mean |diff| {seg0_max:.2e} >= {SEG0_TOL}")
    v2 = run_exp2(tl, docs, a_src, a_doc, b_src, b_doc)
    tl.close()
    if tl.max_dev > 2e-2:
        warnings.append(f"per-token loss deviates from the fused training loss by up to {tl.max_dev:.4f} per forward")

    np.savez(out / "rows.npz", run=run.name, h1=h1, h2=h2, src=src, doc=doc, a_src=a_src, a_doc=a_doc, b_src=b_src,
             b_doc=b_doc, **{f"e1_{c}": v for c, v in v1.items()}, **{f"e2_{c}": v for c, v in v2.items()})
    head = {"run": run.name, "ckpt": ckpt.name, "step": step, "gate_mode": model.c.gate_mode, "seed": args.seed,
            "bucket_edges": EDGES, "bucket_tokens_per_segment": CNT.astype(int).tolist(), "sources": SOURCES}
    st1 = exp1_stats(v1, src, doc)
    E.dump(out, "exp1_switch.json", {
        **head, "definition": DEF_EXP1, "rows": len(src), "spec_hash": h1, "unique_docs": unique_docs(src, doc),
        "check_segment0": {"max_abs_bucket_mean_b_minus_a": seg0_max, "max_abs_token_b_minus_a": seg0_tok,
                           "tol": SEG0_TOL, "ok": seg0_ok},
        "conditions": {"a": "reset at segment starts", "b": "no reset", "b_minus_a": "(b) - (a), paired per observation"},
        "stats": st1}, warnings)
    st2 = exp2_stats([v2], a_src, a_doc, args.seed)
    E.dump(out, "exp2_interference.json", {
        **head, "definition": DEF_EXP2, "rows": len(a_src), "spec_hash": h2, "L": list(LS),
        "unique_docs_A": unique_docs(a_src, a_doc), "unique_docs_B": unique_docs(b_src, b_doc),
        "rows_per_A_source": {s: int((a_src == k).sum()) for k, s in enumerate(SOURCES)},
        "stats": st2, "max_forward_dev_tokenwise_vs_fused": tl.max_dev, "warnings": warnings,
        "seconds": time.time() - t0, "max_mem_gb": torch.cuda.max_memory_allocated(device) / 2 ** 30}, warnings)
    print_exp1(st1, run.name)
    print_exp2(st2, run.name)
    for w in warnings:
        log(f"WARNING: {w}")
    log(f"done {out} {time.time() - t0:.0f}s")


def compare(args):
    ra, rb = (Path(p).resolve() for p in args.compare)
    root = Path(args.out).resolve()
    za, zb = (np.load(root / r.name / "rows.npz") for r in (ra, rb))
    for k in ("h1", "h2"):
        if str(za[k]) != str(zb[k]):
            E.die(f"row specs differ between runs ({k}): rerun both with the same --rows/--seed")
    src, doc, a_src, a_doc = za["src"], za["doc"], za["a_src"], za["a_doc"]
    seed = args.seed
    out = root / f"paired_{rb.name}_minus_{ra.name}"
    out.mkdir(parents=True, exist_ok=True)
    warnings = []
    head = {"A": ra.name, "B": rb.name, "difference": "B - A, paired by row (same rows for both)",
            "bucket_edges": EDGES, "sources": SOURCES}
    v1 = {c: zb[f"e1_{c}"] - za[f"e1_{c}"] for c in ("a", "b", "b_minus_a")}
    st1 = exp1_stats(v1, src, doc)
    E.dump(out, "exp1_switch.json", {**head, "definition": DEF_EXP1, "rows": len(src), "stats": st1}, warnings)
    cond = ["i", "iii"] + [f"ii_{L}" for L in LS]
    m2 = [{c: z[f"e2_{c}"] for c in cond} for z in (za, zb)]
    st2 = exp2_stats(m2, a_src, a_doc, seed)
    E.dump(out, "exp2_interference.json", {**head, "definition": DEF_EXP2 + " Entries are B - A of each quantity; "
                                           "retained_fraction SE from the same bootstrap resamples for both runs.",
                                           "rows": len(a_src), "stats": st2}, warnings)
    print_exp1(st1, f"PAIRED {rb.name} - {ra.name}")
    print_exp2(st2, f"PAIRED {rb.name} - {ra.name}")
    for w in warnings:
        log(f"WARNING: {w}")
    log(f"done {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run directory (model/<run name>)")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="paired B - A from both rows.npz")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "switch"))
    ap.add_argument("--rows", type=int, default=2000, help="rows per experiment")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if bool(args.run) == bool(args.compare):
        ap.error("give exactly one of --run / --compare")
    try:
        compare(args) if args.compare else run_one(args)
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        E.die(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
